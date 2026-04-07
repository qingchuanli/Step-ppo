#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from FlagEmbedding import FlagAutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Distance, VectorParams

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "corpus" / "hotpotqa"


def _stable_int_id(title: str, text: str) -> int:
    """
    生成稳定的 64-bit 正整数 id，保证脚本可幂等重复跑（不会因为 next_id 重置导致重复/错位）。
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(title.encode("utf-8", errors="ignore"))
    h.update(b"\x1f")
    h.update(text.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "big", signed=False)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Qdrant collection for HotpotQA wiki_search.")
    parser.add_argument("--corpus_path", type=str, default=str(_DATA_DIR / "hpqa_corpus.jsonl"))
    parser.add_argument("--db_path", type=str, default=str(_DATA_DIR / "qdrant_db"))
    parser.add_argument("--collection", type=str, default="hpqa_corpus")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--vector_size", type=int, default=1024, help="bge-large-en-v1.5 = 1024")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection (DANGEROUS).")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    client = QdrantClient(path=str(Path(args.db_path)))

    if args.recreate and client.collection_exists(args.collection):
        client.delete_collection(args.collection)

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=int(args.vector_size), distance=Distance.COSINE),
        )

    # 对 title 建 payload 索引，方便精确过滤（可选）
    try:
        client.create_payload_index(args.collection, "title", field_schema="keyword")
    except Exception:
        # index 已存在时可能会报错；忽略即可
        pass

    model = FlagAutoModel.from_finetuned(args.embedding_model)

    batch: list[dict] = []
    for rec in _iter_jsonl(corpus_path):
        title = str(rec.get("title", "")).strip()
        text = str(rec.get("text", "")).strip()
        if not title or not text:
            continue
        batch.append({"title": title, "text": text})
        if len(batch) >= int(args.batch_size):
            _flush(client, model, args.collection, batch)
            batch = []

    if batch:
        _flush(client, model, args.collection, batch)


def _flush(client: QdrantClient, model: FlagAutoModel, collection: str, batch: list[dict]) -> None:
    texts = [f'{d["title"]} {d["text"]}' for d in batch]
    vectors = model.encode_corpus(texts)
    ids = [_stable_int_id(d["title"], d["text"]) for d in batch]
    client.upsert(
        collection_name=collection,
        points=Batch(
            ids=ids,
            vectors=vectors.tolist(),
            payload=batch,
        ),
    )


if __name__ == "__main__":
    main()
