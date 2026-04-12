"""
HotpotQA prompt & tool schema.

Design aligned with Agent-R1-legacy's approach:
- User prompt includes: question + instruction + accumulated passages + action history
- Model output format: <think>...</think> then <tool_call>...</tool_call> or <answer>...</answer>
- Each step re-builds the full prompt from current state (not multi-turn message accumulation),
  keeping prompt length bounded and predictable.
"""

HOTPOTQA_SYSTEM_PROMPT = (
    "You are a multi-hop QA research agent. "
    "You can call the search tool to retrieve Wikipedia passages, then reason and answer."
)

INSTRUCTION_FOLLOWING = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in <answer> </answer> tags."
)

HOTPOTQA_USER_PROMPT = """### Question
{user_query}

### Retrieved Passages
{passage_list}

### History Actions
{history_actions}

### Instructions
{instruction_following}
- You can call the `search` tool with a natural language query to retrieve relevant passages.
- You may call `search` multiple times to gather evidence from different angles.
- When calling search, output in the following format:

<tool_call>
{{"name":"search","arguments":{{"query":"YOUR SEARCH QUERY"}}}}
</tool_call>

- When you have enough evidence, provide the final answer:

<answer>
YOUR FINAL ANSWER HERE
</answer>

Do NOT explain the reasoning in the final answer, only provide the answer text inside the <answer> tags.
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
