import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.hotpotqa.prompts import (
    HOTPOTQA_SYSTEM_PROMPT,
    HOTPOTQA_TOOL_SCHEMAS,
    format_hotpotqa_initial_user,
)
from recipe.hotpotqa.utils import HotpotQASearchToolLegacy, PassagePool, parse_legacy_tool_result
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
LOG_DIR = "/root/data/log"
LOG_PATH = f"{LOG_DIR}/hotpotqa_agent_flow.log"

# Qwen / vLLM 等栈里 function name 可能与 recipe 里 OpenAI schema 的 "search" 不一致（例如 wiki_search）
_RETRIEVAL_TOOL_NAMES = frozenset({"search", "wiki_search"})


def _decode_tool_arguments(arguments: str) -> dict[str, Any] | None:
    """Hermes 里 arguments 应为 JSON object；少数模型会再套一层 JSON 字符串。"""
    try:
        obj: Any = json.loads(arguments)
    except Exception:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


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
    Multi-step HotpotQA agent（多轮对话式上下文）:

    - Prompt 由 system + 累积 user/assistant 构成；每轮保留 assistant 解码原文；
      检索后以 user 轮注入工具 JSON 与可读段落（不再用单块 action 历史拼盘）。
    - Tools: search / wiki_search（本地 FAISS）
    - Reward: 工具步 reward_score = 0.0；最终无 tool call 时 reward_score = None → custom_reward_function（EM）。
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
        # Defaults should match recipe/hotpotqa/base_faiss_cpu.yaml
        self.max_steps = int(kwargs.get("max_steps", 5))
        self.max_parallel_calls = int(kwargs.get("max_parallel_calls", 4))
        self.force_first_search = bool(kwargs.get("force_first_search", True))

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format,
            self.tokenizer,
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = HOTPOTQA_TOOL_SCHEMAS

        # Legacy-compatible local FAISS search tool（路径写死在 recipe/hotpotqa/utils.py）
        embedding_model_name = kwargs.get("embedding_model_name", "BAAI/bge-large-en-v1.5")
        embedding_devices = kwargs.get("embedding_devices", None)
        self.search_tool = HotpotQASearchToolLegacy(
            embedding_model_name=embedding_model_name,
            embedding_devices=embedding_devices,
        )

        self.passage_pool = PassagePool()
        self._dialog_messages: list[dict[str, str]] = []
        self._num_search_rounds: int = 0
        self.question: str = ""
        self.ground_truth: str = ""
        self.steps: list[AgentFlowStep] = []

    @staticmethod
    def _format_passages_readable(tool_content: str) -> str:
        passages = parse_legacy_tool_result(tool_content)
        if not passages:
            return "(No passages in payload or JSON parse failed.)"
        lines: list[str] = []
        for i, p in enumerate(passages, start=1):
            snippet = p.text[:1200].replace("\n", " ")
            lines.append(f"  [{i}] {snippet}")
        return "\n".join(lines)

    def _format_search_observation_message(
        self,
        queries: list[str],
        tool_results: list[dict[str, Any]],
        *,
        footer: bool = True,
    ) -> str:
        """将检索结果格式化为 user 轮内容（含工具原始 JSON + 可读段落）。"""
        if not queries:
            body = (
                "No executable `search` / `wiki_search` call was parsed from your last assistant message "
                "(check tool name and JSON). Use:\n\n"
                '<tool_call>\n{"name":"search","arguments":{"query":"YOUR QUERY"}}\n</tool_call>'
            )
            return body + (
                "\n\nPlease continue: call `search` if needed, or answer in <answer>...</answer>."
                if footer
                else ""
            )

        parts: list[str] = [
            "### Search tool output",
            "Below is what the local search tool returned for your last tool call(s).",
            "",
        ]
        for q, item in zip(queries, tool_results, strict=True):
            parts.append(f"**Query:** {q}")
            parts.append("")
            if not item.get("success", False):
                parts.append(f"(Search failed) {item.get('content', '')}")
            else:
                raw = str(item.get("content", ""))
                parts.append("**Raw JSON (tool `content`):**")
                parts.append(raw)
                parts.append("")
                parts.append("**Passages (readable):**")
                parts.append(self._format_passages_readable(raw))
            parts.append("")
            parts.append("---")
            parts.append("")
        if footer:
            parts.append(
                "Please continue: call `search` again if you need more evidence, "
                "or output the final answer in <answer>...</answer> when ready."
            )
        return "\n".join(parts).rstrip() + "\n"

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
        self._dialog_messages = []
        self._num_search_rounds = 0

        bootstrap_block = ""
        if self.force_first_search:
            bootstrap_block = await self._bootstrap_passages_from_question()

        self._dialog_messages = [
            {
                "role": "user",
                "content": format_hotpotqa_initial_user(self.question, bootstrap_block),
            }
        ]

        try:
            while num_steps < self.max_steps:
                num_steps += 1

                messages = [{"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT}, *self._dialog_messages]

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
                                "num_tool_steps": self._num_search_rounds,
                            },
                        },
                    )
                    step = await self._postprocess(step, **kwargs)
                    self.steps.append(step)
                    break

                # Tool calls exist: only support search, and do not assign reward here.
                tool_calls = tool_calls[: self.max_parallel_calls]

                queries: list[str] = []
                for tool_call in tool_calls:
                    if tool_call.name not in _RETRIEVAL_TOOL_NAMES:
                        continue
                    tool_args = _decode_tool_arguments(tool_call.arguments)
                    if not tool_args:
                        logger.warning("Failed to parse tool arguments for %s", tool_call.name)
                        continue
                    query = tool_args.get("query")
                    if not query:
                        continue
                    queries.append(str(query))

                new_passages_list: list[list[Any]] = [[] for _ in queries]
                tool_results: list[dict[str, Any]] = []
                if queries:
                    with simple_timer("tool_calls", metrics):
                        try:
                            # 不在默认 ThreadPoolExecutor 里跑：CUDA/BGE 在非主线程常失败且难排查
                            tool_results = self.search_tool.batch_execute([{"query": q} for q in queries])
                            parsed_results: list[list[Any]] = []
                            for item in tool_results:
                                if not item.get("success", False):
                                    parsed_results.append([])
                                    continue
                                parsed_results.append(parse_legacy_tool_result(str(item.get("content", ""))))
                            new_passages_list = parsed_results
                        except Exception as e:
                            logger.warning(
                                "[hotpotqa_agent][trajectory=%s] batch search failed, err=%s",
                                trajectory_uid,
                                e,
                            )
                            new_passages_list = [[] for _ in queries]
                            tool_results = [{"content": str(e), "success": False} for _ in queries]

                # 多轮对话：保留本轮 assistant 原文，再把检索结果作为下一轮 user 内容写入历史
                self._dialog_messages.append({"role": "assistant", "content": response_text})
                obs = self._format_search_observation_message(queries, tool_results, footer=True)
                self._dialog_messages.append({"role": "user", "content": obs})

                for query, passages in zip(queries, new_passages_list, strict=False):
                    for p in passages:
                        self.passage_pool.add_passage(p)
                if queries:
                    self._num_search_rounds += len(queries)

                # Tool step: reward_score=0.0 (no shaping), but we still push step into trajectory.
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=0.0,
                    extra_fields={
                        "reward_extra_info": {
                            "num_tool_steps": self._num_search_rounds,
                        },
                    },
                )
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
        finally:
            pass
        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    async def _bootstrap_passages_from_question(self) -> str:
        """执行首轮隐式检索，填充 passage_pool，并返回写入首轮 user 的检索文本（无结果则空串）。"""
        if not (self.question or "").strip():
            return ""
        try:
            tool_results = self.search_tool.batch_execute([{"query": self.question}])
            added = 0
            for item in tool_results:
                if not item.get("success", False):
                    continue
                for p in parse_legacy_tool_result(str(item.get("content", ""))):
                    before = len(self.passage_pool.passages)
                    self.passage_pool.add_passage(p)
                    if len(self.passage_pool.passages) > before:
                        added += 1
            if not added:
                return ""
            self._num_search_rounds += 1
            return self._format_search_observation_message(
                [self.question],
                tool_results,
                footer=False,
            )
        except Exception as e:
            logger.warning("[hotpotqa_agent] bootstrap search failed: %s", e)
            return ""