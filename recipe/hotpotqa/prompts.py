"""
HotpotQA prompt & tool schema.

Tool call format is handled entirely by the chat template (via `tools=` parameter
in apply_chat_template). The user prompt should NOT include explicit <tool_call> JSON
format — doing so conflicts with the template-generated instructions and causes the
model to output malformed JSON.

This follows the same pattern as recipe/paper_search.
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

### Instructions
Analyze the retrieved passages and history actions, then decide your next step.
You FIRST think about the reasoning process as an internal monologue and then take action.
- You may call the `search` tool one or more times to retrieve relevant Wikipedia passages.
- Attend to the history actions and avoid repeating the same queries.
- When you have gathered enough evidence, provide the final answer inside <answer> </answer> tags.

### Output Format
<tool_call>
[search tool call]
</tool_call>

OR when ready to answer:

<answer>
[Your final answer here — short and precise, no explanation]
</answer>
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
