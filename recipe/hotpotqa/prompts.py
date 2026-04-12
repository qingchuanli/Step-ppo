HOTPOTQA_SYSTEM_PROMPT = (
    "You are a multi-hop QA research agent. "
    "You can call tools to search a local Wikipedia corpus to answer the question. "
    "Use tools to gather enough evidence, then give the final answer. "
    "The conversation below is multi-turn: your previous replies are shown verbatim; "
    "after each search, the user message contains the tool output for you to read."
)


# 首轮 user 仅含任务说明；检索结果通过后续 user 轮次注入（见 agent_flow 拼消息）
HOTPOTQA_INSTRUCTIONS = """### Instructions
- You can call the `search` tool with a natural language query to retrieve relevant passages.
- You may call `search` multiple times to gather evidence.
- If more evidence is needed, output a tool call in the exact format below (Hermes parser):

<tool_call>
{{"name":"search","arguments":{{"query":"YOUR SEARCH QUERY"}}}}
</tool_call>

- After you are confident, STOP calling tools and directly output the final answer in the following format:

<answer>
YOUR FINAL ANSWER HERE
</answer>

Do NOT explain the reasoning in the final answer, only provide the answer text inside the <answer> tags.
"""


def format_hotpotqa_initial_user(user_query: str, initial_retrieval_block: str = "") -> str:
    """
    构造首轮 user 内容。若 `force_first_search` 已有 bootstrap 段落，传入 `initial_retrieval_block`（纯文本）。
    """
    extra = ""
    if (initial_retrieval_block or "").strip():
        extra = f"\n### Initial retrieval\n{initial_retrieval_block.strip()}\n"
    return f"""### Question
{user_query}
{extra}
{HOTPOTQA_INSTRUCTIONS}"""


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search for relevant Wikipedia passages from a local HotpotQA "
            "corpus using semantic retrieval with FAISS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query describing what you want to find.",
                },
            },
            "required": ["query"],
        },
    },
}


HOTPOTQA_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA]