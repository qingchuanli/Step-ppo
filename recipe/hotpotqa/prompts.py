"""
HotpotQA prompt & tool schema.

Tool call format is handled entirely by the chat template (via `tools=` parameter
in apply_chat_template). The user prompt must NOT contain any <tool_call> tags or
JSON format examples — they conflict with the template-generated instructions and
cause the model to output malformed JSON.

Key rule (learned from Agent-R1-legacy):
  - Agent-R1-legacy: apply_chat_template(messages, tools=self.tools) generates tool
    format in system area; user prompt contains ONLY the question and task instruction.
  - Paper_search: same pattern, user prompt mentions tool calls generically but
    Qwen3-4B handles the dual instructions fine; Qwen2.5-3B does NOT.
  - Therefore: for Qwen2.5, user prompt must have ZERO *example* <tool_call> blocks (only dynamic feedback below).
"""

HOTPOTQA_SYSTEM_PROMPT = (
    "You are a multi-hop QA research agent. "
    "You can call the search tool to retrieve Wikipedia passages, then reason and answer."
)

HOTPOTQA_USER_PROMPT = """### Question
{user_query}

### Retrieved Passages
{passage_list}

### History Actions
{history_actions}

### Recent tool / format issues
{tool_feedback}

### Instructions
Analyze the retrieved passages and history actions, then decide your next step.
- You may call the `search` tool to retrieve relevant Wikipedia passages.
- You may call `search` multiple times to gather evidence from different angles.
- Attend to the history actions and avoid repeating the same queries.
- When you have gathered enough evidence, provide the final answer inside <answer> </answer> tags. The answer should be short and precise, with no explanation.
"""

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information on the internet using Wikipedia as a knowledge source.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}

HOTPOTQA_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA]
