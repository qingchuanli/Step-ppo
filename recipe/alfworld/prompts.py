ALFWORLD_SYSTEM_PROMPT = (
    "You are an agent in a text-based household environment (ALFWorld). "
    "You interact with the environment ONLY by issuing single-line commands "
    "such as 'go north', 'open fridge', 'take apple', 'put apple in fridge'. "
    "Do not explain your reasoning in the command itself."
)


ALFWORLD_USER_PROMPT = """### Current Observation
{observation}

### History Actions
{history_actions}

### Goal
{goal}

### Instructions
- At each step, output exactly ONE command via the `env_step` tool.
- Commands should be concise and executable in the environment.
- Try to accomplish the goal in as few steps as possible.
- Do NOT output free-form text answers; always use the tool.
"""


EXEC_ACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": (
            "Execute a single text command in the ALFWorld environment. "
            "The command should be a valid action like 'go north', "
            "'open fridge', 'take apple', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single text command to execute in the environment.",
                }
            },
            "required": ["command"],
        },
    },
}


ALFWORLD_TOOL_SCHEMAS = [EXEC_ACTION_TOOL_SCHEMA]

