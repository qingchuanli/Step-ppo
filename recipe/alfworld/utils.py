from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from recipe.alfworld.env.alfworld_wrapper import AlfworldEnvWrapper


@dataclass
class AlfworldToolExecutor:
    """
    Simple executor that bridges tool calls from AgentFlow to ALFWorld env.
    """

    env_name: str = "alfworld_train"
    _env: AlfworldEnvWrapper = field(init=False)
    _history_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._env = AlfworldEnvWrapper(env_name=self.env_name)

    def reset(self, task_id: str | None = None) -> str:
        self._history_actions.clear()
        return self._env.reset(task_id=task_id)

    def step(self, command: str) -> Dict[str, Any]:
        self._history_actions.append(command)
        observation, reward, done, info = self._env.step(command)
        return {
            "observation": str(observation),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "history_actions": list(self._history_actions),
        }


def format_history_actions(actions: List[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Action {i + 1}] {a}" for i, a in enumerate(actions))

