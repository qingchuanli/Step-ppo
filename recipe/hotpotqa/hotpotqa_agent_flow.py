import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.hotpotqa.prompts import HOTPOTQA_SYSTEM_PROMPT, HOTPOTQA_TOOL_SCHEMAS, HOTPOTQA_USER_PROMPT
from recipe.hotpotqa.utils import PassagePool, WikiQdrantRetriever, format_history_actions
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
LOG_DIR = "/root/data/log"
LOG_PATH = f"{LOG_DIR}/hotpotqa_agent_flow.log"


def _ensure_file_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    if any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == LOG_PATH
        for handler in logger.handlers
    ):
        return
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.WARN)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)


@register("hotpotqa_agent")
class HotpotQAAgentFlow(AgentFlowBase):
    """
    Multi-step HotpotQA agent:

    - Tools: wiki_search (local Qdrant)
    - Reward:
      * All tool steps: reward_score = 0.0 (no shaping)
      * Final step (no tool calls): reward_score = None,
        so ARFT RewardLoop will call custom_reward_function (exact match EM).
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, reward_loop_worker, tokenizer, processor, **kwargs)
        _ensure_file_logger()
        self.max_steps = kwargs.get("max_steps", 5)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 4)

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = HOTPOTQA_TOOL_SCHEMAS

        # Qdrant wiki retriever (local DB path is resolved inside the class by default).
        qdrant_path = kwargs.get("qdrant_path", None)
        qdrant_url = kwargs.get("qdrant_url", None)
        collection_name = kwargs.get("collection_name", "hpqa_corpus")
        embedding_model_name = kwargs.get("embedding_model_name", "BAAI/bge-large-en-v1.5")
        self.retriever = WikiQdrantRetriever(
            db_path=qdrant_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
        )

        self.passage_pool = PassagePool()
        self.history_actions: list[tuple[str, str]] = []
        self.question: str = ""
        self.ground_truth: str = ""
        self.steps: list[AgentFlowStep] = []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        # Dataset gives prompt as [{"role": "user", "content": question}, ...]
        self.question = raw_prompt[0]["content"]
        trajectory_uid = kwargs.get("uid", "unknown")

        # ground truth is stored in non_tensor_batch["reward_model"]["ground_truth"]
        reward_model = kwargs.get("reward_model") or {}
        self.ground_truth = str(reward_model.get("ground_truth", "")).strip()

        metrics: dict[str, Any] = {}
        num_steps = 0
        self.steps = []
        self.passage_pool = PassagePool()
        self.history_actions = []

        try:
            while num_steps < self.max_steps:
                num_steps += 1

                messages = [
                    {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": HOTPOTQA_USER_PROMPT.format(
                            user_query=self.question,
                            passage_list=self.passage_pool.passage_list,
                            history_actions=format_history_actions(self.history_actions),
                        ),
                    },
                ]

                prompt_ids = await self.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                )

                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                    )

                response_ids = output.token_ids[: self.response_length]
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                logger.warning(
                    (
                        "[hotpotqa_agent][trajectory=%s][step=%d] INPUT_TEXT:\n%s\n"
                        "[hotpotqa_agent][trajectory=%s][step=%d] OUTPUT_TEXT:\n%s"
                    ),
                    trajectory_uid,
                    num_steps,
                    prompt_text,
                    trajectory_uid,
                    num_steps,
                    response_text,
                )
                _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

                # No tool calls: treat as final answer step.
                if not tool_calls:
                    step = AgentFlowStep(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                        # IMPORTANT:
                        # reward_score=None so ARFT RewardLoopWorker will call custom_reward_function
                        # (exact match EM) based on decoded response + ground_truth from dataset.
                        reward_score=None,
                        extra_fields={
                            "reward_extra_info": {
                                "num_tool_steps": len(self.history_actions),
                            },
                        },
                    )
                    step = await self._postprocess(step, **kwargs)
                    self.steps.append(step)
                    break

                # Tool calls exist: only support wiki_search, and do not assign reward here.
                tool_calls = tool_calls[: self.max_parallel_calls]

                tasks = []
                queries: list[str] = []
                for tool_call in tool_calls:
                    try:
                        tool_args = json.loads(tool_call.arguments)
                    except Exception as e:
                        logger.warning("Failed to parse tool arguments: %s", e)
                        continue

                    if tool_call.name == "wiki_search":
                        query = tool_args.get("query")
                        if not query:
                            continue
                        top_k = int(tool_args.get("top_k", 5))
                        queries.append(query)
                        # Run blocking retriever.search in thread pool.
                        tasks.append(
                            asyncio.get_running_loop().run_in_executor(
                                None,
                                self.retriever.search,
                                query,
                                top_k,
                            )
                        )

                new_passages_list: list[list[Any]] = []
                if tasks:
                    with simple_timer("tool_calls", metrics):
                        new_passages_list = await asyncio.gather(*tasks)

                # Update passage pool & history
                for query, passages in zip(queries, new_passages_list, strict=False):
                    self.history_actions.append(("search", query))
                    for p in passages:
                        self.passage_pool.add_passage(p)

                # Tool step: reward_score=0.0 (no shaping), but we still push step into trajectory.
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=0.0,
                    extra_fields={
                        "reward_extra_info": {
                            "num_tool_steps": len(self.history_actions),
                        },
                    },
                )
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
        finally:
            # Avoid FlagEmbedding multiprocess pool teardown during interpreter shutdown (__del__).
            self.retriever.close()

        return AgentFlowOutput(steps=self.steps, metrics=metrics)