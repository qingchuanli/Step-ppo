from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Custom reward for ALFWorld.

    - If data_source != "alfworld": fall back to default_compute_score so other
      datasets are unaffected.
    - If data_source == "alfworld":
        * We only care about final success/fail.
        * ground_truth is expected to be a bool or {"success": bool}, or we can
          read success flag from extra_info.
    """
    if data_source != "alfworld":
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    success_flag: bool | None = None

    # Prefer explicit ground_truth if provided
    if isinstance(ground_truth, dict) and "success" in ground_truth:
        success_flag = bool(ground_truth["success"])
    elif isinstance(ground_truth, bool):
        success_flag = ground_truth

    # Fallback to extra_info if available
    if success_flag is None and extra_info is not None:
        if isinstance(extra_info, dict) and "success" in extra_info:
            success_flag = bool(extra_info["success"])

    if success_flag is None:
        # Unknown success state -> neutral reward
        return 0.0

    return 1.0 if success_flag else 0.0

