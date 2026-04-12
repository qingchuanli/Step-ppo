import re
import string
from typing import Any

from verl.utils.reward_score import default_compute_score


def _normalize_answer(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _extract_answer_from_solution(solution_str: str) -> str:
    """
    Prefer content inside <answer>...</answer>. If not present, fall back to full string.
    """
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, solution_str, flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return solution_str.strip()
    return matches[-1].group(1).strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Custom reward function for HotpotQA.

    - If data_source == "hotpotqa_distractor": use simple exact match (EM) between predicted
      answer and ground_truth string.
    - Otherwise, fall back to verl's default_compute_score.
    """
    if data_source != "hotpotqa_distractor":
        # Delegate to built-in reward logic for other datasets if any.
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    if ground_truth is None:
        return 0.0

    gt_str = str(ground_truth).strip()
    if not gt_str:
        return 0.0

    pred = _extract_answer_from_solution(solution_str or "")
    norm_pred = _normalize_answer(pred)
    norm_gt = _normalize_answer(gt_str)

    return 1.0 if norm_pred == norm_gt else 0.0

