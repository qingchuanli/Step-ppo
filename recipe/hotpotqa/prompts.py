"""
HotpotQA prompt & tool schema.

Rollout uses `apply_chat_template(..., tools=HOTPOTQA_TOOL_SCHEMAS)` **and** an explicit
user-side output contract, matching `recipe/paper_search/prompts.py`: the chat template
injects schemas, but models still need concrete `<tool_call>` + JSON shape guidance to
produce strings that `HermesToolParser` / env can parse.

The JSON inside each `<tool_call>...</tool_call>` must be parseable as:
  {"name": "search", "arguments": {"query": "<string>"}}
with double-quoted keys and string values (no single quotes, no trailing commas).
"""

HOTPOTQA_SYSTEM_PROMPT = (
    "You are a multi-hop QA research agent. "
    "You retrieve Wikipedia evidence with the `search` tool using the required assistant format below, "
    "then answer concisely when you have enough support."
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
Read the passages and history. Decide whether you need **new** retrieval or can answer from current evidence.
- If you need more evidence, call `search` (do not repeat identical queries from history).
- If evidence is sufficient, answer without calling `search`.

### Output format (required for rollout)
Put brief reasoning inside `<analysis>...</analysis>` if helpful, then either tool calls or the final answer.

**To call search** — emit one or more blocks exactly like this (only double quotes inside JSON; `name` must be `search`):

<tool_call>
{{"name": "search", "arguments": {{"query": "your concise English search query"}}}}
</tool_call>

You may emit **multiple** `<tool_call>...</tool_call>` blocks in one turn for parallel searches.

**To finish** — when no further search is needed, output only:

<answer>short exact answer, no explanation</answer>
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
