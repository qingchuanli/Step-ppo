"""
HotpotQA AgentFlow — multi-step search agent for multi-hop QA.

Architecture follows recipe/paper_search style:
- Each step re-builds messages from current state (not multi-turn message accumulation)
- Passages and action history are maintained as structured state, rendered into prompt each step
- Prompt length is bounded: passages are truncated to fit within budget
- Tool call format is handled by chat template (via tools= in apply_chat_template),
  NOT by manual format instructions in the user prompt
- Search tool backed by local FAISS + BGE (HotpotQASearchToolLegacy)
- Reward: tool steps get reward_score=0.0; final step gets reward_score=None (→ custom EM reward)
"""

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
    HOTPOTQA_USER_PROMPT,
)
from recipe.hotpotqa.utils import HotpotQASearchToolLegacy, parse_legacy_tool_result
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_RETRIEVAL_TOOL_NAMES = frozenset({"search", "wiki_search"})


def _decode_tool_arguments(arguments: str) -> dict[str, Any] | None:
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


def _format_passage_list(passages: list[tuple[str, str]], max_chars: int = 0) -> str:
    """Format accumulated passages for prompt. Each entry is (query, text)."""
    if not passages:
        return "None"
    lines: list[str] = []
    total = 0
    for i, (query, text) in enumerate(passages, start=1):
        snippet = text[:1200].replace("\n", " ")
        line = f"[{i}] (query: {query}) {snippet}"
        if max_chars > 0 and total + len(line) > max_chars:
            lines.append(f"... ({len(passages) - i + 1} more passages truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search] {q}" for q in actions)


@register("hotpotqa_agent")
class HotpotQAAgentFlow(AgentFlowBase):
    """
    Multi-step HotpotQA agent (paper_search style state management).

    Each step re-builds [system, user] from current state:
    - user_query (fixed)
    - passage_list (accumulated, truncated to fit)
    - history_actions (list of past queries)
    This avoids prompt length explosion from multi-turn message accumulation.
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

        embedding_model_name = kwargs.get("embedding_model_name", "BAAI/bge-large-en-v1.5")
        embedding_devices = kwargs.get("embedding_devices", None)
        self.search_tool = HotpotQASearchToolLegacy(
            embedding_model_name=embedding_model_name,
            embedding_devices=embedding_devices,
        )

    def _build_messages(self, question: str, passages: list[tuple[str, str]], actions: list[str]) -> list[dict]:
        """Build [system, user] messages from current state, with passage truncation for safety."""
        max_passage_chars = self.prompt_length * 3
        user_content = HOTPOTQA_USER_PROMPT.format(
            user_query=question,
            passage_list=_format_passage_list(passages, max_chars=max_passage_chars),
            history_actions=_format_history_actions(actions),
        )
        return [
            {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _make_extra_fields(self, history_actions: list[str], acc: float = 0.0) -> dict[str, Any]:
        """Build extra_fields with consistent reward_extra_info keys across all steps."""
        return {
            "reward_extra_info": {
                "num_tool_steps": len(history_actions),
                "acc": acc,
            },
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = raw_prompt[0]["content"]

        metrics: dict[str, Any] = {}
        steps: list[AgentFlowStep] = []
        passages: list[tuple[str, str]] = []
        history_actions: list[str] = []

        if self.force_first_search:
            self._do_search(question, passages, history_actions)

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1

            messages = self._build_messages(question, passages, history_actions)
            prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)

            if len(prompt_ids) > self.prompt_length:
                logger.warning(
                    "[hotpotqa_agent][step=%d] prompt too long (%d tokens, limit %d); truncating.",
                    num_steps, len(prompt_ids), self.prompt_length,
                )
                prompt_ids = prompt_ids[-self.prompt_length:]

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=None,
                    extra_fields=self._make_extra_fields(history_actions),
                )
                step = await self._postprocess(step, **kwargs)
                ri = step.extra_fields.get("reward_extra_info", {})
                step.extra_fields["reward_extra_info"] = {
                    "num_tool_steps": len(history_actions),
                    "acc": ri.get("acc", 0.0),
                }
                steps.append(step)
                break

            tool_calls = tool_calls[: self.max_parallel_calls]

            queries: list[str] = []
            for tc in tool_calls:
                if tc.name not in _RETRIEVAL_TOOL_NAMES:
                    continue
                tool_args = _decode_tool_arguments(tc.arguments)
                if not tool_args:
                    continue
                query = tool_args.get("query")
                if query:
                    queries.append(str(query))

            if queries:
                with simple_timer("tool_calls", metrics):
                    self._do_search_batch(queries, passages, history_actions)

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=0.0,
                extra_fields=self._make_extra_fields(history_actions),
            )
            step = await self._postprocess(step, **kwargs)
            steps.append(step)

        return AgentFlowOutput(steps=steps, metrics=metrics)

    def _do_search(self, query: str, passages: list[tuple[str, str]], history_actions: list[str]) -> None:
        """Execute a single search query and update state."""
        try:
            results = self.search_tool.batch_execute([{"query": query}])
            self._ingest_results(query, results, passages)
            history_actions.append(query)
        except Exception as e:
            logger.warning("[hotpotqa_agent] search failed for query=%r: %s", query, e)
            history_actions.append(query)

    def _do_search_batch(self, queries: list[str], passages: list[tuple[str, str]], history_actions: list[str]) -> None:
        """Execute multiple search queries and update state."""
        try:
            results = self.search_tool.batch_execute([{"query": q} for q in queries])
            for query, item in zip(queries, results):
                self._ingest_results(query, [item], passages)
                history_actions.append(query)
        except Exception as e:
            logger.warning("[hotpotqa_agent] batch search failed: %s", e)
            for q in queries:
                history_actions.append(q)

    @staticmethod
    def _ingest_results(
        query: str,
        results: list[dict[str, Any]],
        passages: list[tuple[str, str]],
    ) -> None:
        """Parse search results and deduplicate into the passage list."""
        for item in results:
            if not item.get("success", False):
                continue
            content = str(item.get("content", ""))
            for p in parse_legacy_tool_result(content):
                if not any(existing_text == p.text for _, existing_text in passages):
                    passages.append((query, p.text))
