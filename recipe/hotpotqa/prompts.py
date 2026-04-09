HOTPOTQA_SYSTEM_PROMPT = (
    "You are a multi-hop QA research agent. "
    "You can call tools to search a local Wikipedia corpus to answer the question. "
    "Use tools to gather enough evidence, then give the final answer."
)


HOTPOTQA_USER_PROMPT = """### Question
{user_query}

### Retrieved Passages
{passage_list}

### History Actions
{history_actions}

### Instructions
- You can call the `wiki_search` tool with a natural language query to retrieve relevant passages.
- You may call `wiki_search` multiple times to gather evidence.
- If more evidence is needed, output a tool call in the exact format below (Hermes parser):

<tool_call>
{{"name":"wiki_search","arguments":{{"query":"YOUR SEARCH QUERY","top_k":5}}}}
</tool_call>

- After you are confident, STOP calling tools and directly output the final answer in the following format:

<answer>
YOUR FINAL ANSWER HERE
</answer>

Do NOT explain the reasoning in the final answer, only provide the answer text inside the <answer> tags.
"""


WIKI_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki_search",
        "description": (
            "Search for relevant Wikipedia passages from a local HotpotQA "
            "corpus using semantic retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query describing what you want to find.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of passages to retrieve.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        },
    },
}


HOTPOTQA_TOOL_SCHEMAS = [WIKI_SEARCH_TOOL_SCHEMA]