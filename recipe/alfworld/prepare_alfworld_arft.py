#!/usr/bin/env python3
"""
Prepare ALFWorld tasks into VERL/ARFT RLHFDataset-compatible parquet files.

This script does NOT depend on ALFWorld runtime; it only converts task metadata
into a simple prompt-style dataset:

- prompt: [{"role": "user", "content": instruction_text}]
- reward_model: {"ground_truth": {"success": true}}  # placeholder, env gives real success at runtime
- data_source: "alfworld"
- extra_info: {"task_id": ..., "split": ..., ...}

You should point --input_json to a JSONL/JSON file that contains a list of tasks,
each with at least:
  - "task_id": unique id for the task
  - "description": textual goal / instruction
Optionally, you can include:
  - "difficulty", "gamefile", etc. which will be copied into extra_info.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_tasks(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")

    if p.suffix == ".jsonl":
        tasks: list[dict[str, Any]] = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(json.loads(line))
        return tasks
    if p.suffix == ".json":
        with p.open() as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "tasks" in obj:
            return list(obj["tasks"])
        raise ValueError(f"Unsupported JSON structure in {p}")

    raise ValueError(f"Unsupported task file format: {p.suffix}")


def tasks_to_parquet(
    tasks: list[dict[str, Any]],
    split: str,
    output_path: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", f"{split}_{idx}")
        desc = task.get("description") or task.get("goal") or ""
        desc = str(desc).strip()
        if not desc:
            # Skip empty-description tasks
            continue

        instruction = (
            "You are an agent in a text-based household environment. "
            "Your goal is:\n"
            f"{desc}\n\n"
            "Interact step by step by issuing single-line commands (e.g., "
            "'go north', 'open fridge', 'take apple'). "
            "Try to complete the goal as efficiently as possible."
        )

        prompt = [{"role": "user", "content": instruction}]

        # At dataset-prep time we may not know success labels; env will decide at runtime.
        reward_model = {"ground_truth": {"success": None}}

        extra_info = {
            "task_id": task_id,
            "split": split,
        }
        for key in ("difficulty", "gamefile", "room", "seed"):
            if key in task:
                extra_info[key] = task[key]

        row = {
            "data_source": "alfworld",
            "prompt": prompt,
            "reward_model": reward_model,
            "extra_info": extra_info,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ALFWorld tasks into RLHFDataset parquet format.")
    parser.add_argument(
        "--train_tasks",
        type=str,
        required=True,
        help="Path to ALFWorld train tasks file (json or jsonl).",
    )
    parser.add_argument(
        "--val_tasks",
        type=str,
        required=False,
        help="Path to ALFWorld validation tasks file (json or jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/corpus/alfworld",
        help="Output directory for parquet files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tasks = load_tasks(args.train_tasks)
    tasks_to_parquet(train_tasks, "train", str(out_dir / "train.parquet"))

    if args.val_tasks:
        val_tasks = load_tasks(args.val_tasks)
        tasks_to_parquet(val_tasks, "validation", str(out_dir / "validation.parquet"))


if __name__ == "__main__":
    main()

