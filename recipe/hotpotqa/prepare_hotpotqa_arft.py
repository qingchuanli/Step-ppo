#!/usr/bin/env python3
# Copyright 2025 PaperScout / recipe contributors
#
# Download HotpotQA (distractor setting) and export:
# 1) train.parquet / validation.parquet for verl/arft RLHFDataset (prompt + reward_model + data_source)
# 2) hpqa_corpus.jsonl — deduplicated wiki paragraphs from all contexts (for FAISS indexing, see process_hotpotqa.py)
#
# Usage:
#   pip install datasets pyarrow pandas
#   python recipe/hotpotqa/prepare_hotpotqa_arft.py --output_dir data/corpus/hotpotqa

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(
        "Please install dependencies: pip install datasets pyarrow pandas"
    ) from e


def _row_to_arft(
    ex: dict[str, Any],
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Single HotpotQA example -> verl RLHFDataset row."""
    qid = ex.get("_id", f"{split}_{row_index}")
    question = ex["question"].strip()
    answer = ex["answer"]
    if not isinstance(answer, str):
        answer = str(answer)

    # Match paper_search-style rollout: first user message is the task text (agent reads raw_prompt[0]["content"]).
    prompt = [{"role": "user", "content": question}]

    reward_model = {
        "ground_truth": answer,
        "style": "rule",
    }

    extra_info: dict[str, Any] = {
        "index": row_index,
        "question_id": qid,
        "split": split,
        "type": ex.get("type"),
        "level": ex.get("level"),
    }

    return {
        "data_source": "hotpotqa_distractor",
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _iter_context_paragraphs(ex: dict[str, Any]):
    """Yield (title, sentences_list) for one HotpotQA example.

    HuggingFace `hotpot_qa` uses context: {title: [...], sentences: [[...], ...]}.
    Official JSON dumps use context: [[title, [sent, ...]], ...].
    """
    ctx = ex.get("context") or []
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        titles = ctx["title"]
        sents_block = ctx["sentences"]
        for title, sents in zip(titles, sents_block):
            yield title, sents
    else:
        for item in ctx:
            title, sents = item[0], item[1]
            yield title, sents


def _contexts_to_corpus_entries(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten HotpotQA context paragraphs into {title, text} records."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for ex in examples:
        for title, sents in _iter_context_paragraphs(ex):
            text = " ".join(sents).strip()
            title = str(title).strip()
            key = (title, text)
            if not text or key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "text": text})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA for ARFT / verl RL training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/corpus/hotpotqa",
        help="Directory for parquet + hpqa_corpus.jsonl (created if missing).",
    )
    parser.add_argument(
        "--hf_name",
        type=str,
        default="hotpot_qa",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="distractor",
        help="HuggingFace config (distractor = 10 paragraphs per question).",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=-1,
        help="If >0, only keep first N training examples (debug).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=-1,
        help="If >0, only keep first N validation examples (debug).",
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="Do not write hpqa_corpus.jsonl.",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.hf_name} / {args.hf_config} from HuggingFace...")
    ds = load_dataset(args.hf_name, args.hf_config)

    for split_name, parquet_name in [("train", "train.parquet"), ("validation", "validation.parquet")]:
        if split_name not in ds:
            print(f"Skip split {split_name} (not in dataset).")
            continue
        split = ds[split_name]
        rows = []
        max_n = args.max_train if split_name == "train" else args.max_val
        n = len(split) if max_n <= 0 else min(len(split), max_n)
        for i in range(n):
            rows.append(_row_to_arft(split[i], split_name, i))
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, parquet_name)
        df.to_parquet(path, index=False)
        print(f"Wrote {n} rows -> {path}")

    if not args.skip_corpus:
        corpus_path = os.path.join(out_dir, "hpqa_corpus.jsonl")
        all_examples: list[dict[str, Any]] = []
        for split_name, max_n in (
            ("train", args.max_train),
            ("validation", args.max_val),
        ):
            if split_name not in ds:
                continue
            sp = ds[split_name]
            n = len(sp) if max_n <= 0 else min(len(sp), max_n)
            for i in range(n):
                all_examples.append(sp[i])

        entries = _contexts_to_corpus_entries(all_examples)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for rec in entries:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} deduplicated paragraphs -> {corpus_path}")


if __name__ == "__main__":
    main()
