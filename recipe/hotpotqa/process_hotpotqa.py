#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
from FlagEmbedding import FlagAutoModel
import numpy as np

_DEFAULT_DATA_DIR = Path("data/corpus/hotpotqa")


def _load_corpus_texts(corpus_path: Path) -> list[str]:
    corpus: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            corpus.append(f'{rec.get("title", "")} {rec.get("text", "")}'.strip())
    return corpus

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HotpotQA FAISS index (legacy-compatible).")
    parser.add_argument("--data_dir", type=str, default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument(
        "--query_instruction",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus_path) if args.corpus_path else (data_dir / "hpqa_corpus.jsonl")
    emb_path = data_dir / "hpqa_corpus.npy"
    index_path = data_dir / "index.bin"

    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    os.makedirs(str(data_dir), exist_ok=True)
    corpus = _load_corpus_texts(corpus_path)

    model = FlagAutoModel.from_finetuned(
        args.embedding_model,
        query_instruction_for_retrieval=args.query_instruction,
        devices="cpu",
    )
    vectors = model.encode_corpus(corpus)
    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(str(emb_path), vectors)

    corpus_numpy = np.load(str(emb_path)).astype(np.float32)
    dim = corpus_numpy.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, str(index_path))
    print(f"[hotpotqa] saved embeddings to {emb_path}")
    print(f"[hotpotqa] saved index to {index_path}")

if __name__ == "__main__":
    main()