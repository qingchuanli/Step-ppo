"""
Thin wrapper around the ALFWorld text-based environment.

This module assumes you have installed ALFWorld in your environment, e.g.:

    git clone https://github.com/alfworld/alfworld.git
    pip install -e alfworld

The exact import path may vary depending on ALFWorld version. Adjust
`_make_alfworld_env` accordingly if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


def _make_alfworld_env(env_name: str):
    """
    Factory for ALFWorld environment.

    By default we try to import ALFWorld's gym wrapper. If your version uses a
    different entry-point, modify this function.
    """
    try:
        import alfworld.agents.environment as alfworld_env  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - runtime dependency
        raise ImportError(
            "ALFWorld is not installed. Please install it from "
            "https://github.com/alfworld/alfworld and ensure it is on PYTHONPATH."
        ) from e

    # This is roughly aligned with ALFWorld's official usage:
    # env = alfworld_env.ALFWorldEnv(env_name=env_name, ...)
    # For simplicity we pass only env_name; extend as needed.
    env = alfworld_env.ALFWorldEnv(env_name=env_name)
    return env


@dataclass
class AlfworldEnvWrapper:
    """
    A minimal wrapper providing reset/step API used by AlfworldToolExecutor.
    """

    env_name: str = "alfworld_train"

    def __post_init__(self):
        self._env = _make_alfworld_env(self.env_name)
        self._last_obs: str | None = None

    def reset(self, task_id: str | None = None) -> str:
        """
        Reset environment. If task_id is provided and your ALFWorld version
        supports specifying a task, you can route it here.
        """
        if task_id is not None and hasattr(self._env, "reset_task"):
            obs, info = self._env.reset_task(task_id)  # type: ignore[attr-defined]
        else:
            obs, info = self._env.reset()
        self._last_obs = str(obs)
        return self._last_obs

    def step(self, action_str: str) -> Tuple[str, float, bool, dict]:
        """
        Execute a text command in ALFWorld.
        """
        obs, reward, done, info = self._env.step(action_str)
        self._last_obs = str(obs)
        return self._last_obs, float(reward), bool(done), dict(info or {})

