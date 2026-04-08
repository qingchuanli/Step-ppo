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

# 路径配置：根据你的 Docker 映射环境
_DATA_DIR = "/root/data/"

def _stable_int_id(title: str, text: str) -> int:
    """生成稳定的 64-bit 正整数 ID，保证幂等性"""
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
    parser = argparse.ArgumentParser(description="Build Qdrant collection via Docker API.")
    parser.add_argument("--corpus_path", type=str, default=str(_DATA_DIR + "hpqa_corpus.jsonl"))
    # 注意：这里改为 Qdrant 容器的 URL 访问地址
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection", type=str, default="hpqa_corpus")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--vector_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true", help="强制重建集合")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    # 【关键修改】连接到 Docker 运行的 Qdrant 服务
    print(f"正在连接 Qdrant 服务: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    # 处理集合重建逻辑
    if args.recreate and client.collection_exists(args.collection):
        print(f"正在删除旧集合: {args.collection}")
        client.delete_collection(args.collection)

    if not client.collection_exists(args.collection):
        print(f"正在创建新集合: {args.collection}")
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=int(args.vector_size), distance=Distance.COSINE),
        )

    # 加载 Embedding 模型
    print(f"加载模型: {args.embedding_model}")
    model = FlagAutoModel.from_finetuned(args.embedding_model)

    batch: list[dict] = []
    count = 0
    for rec in _iter_jsonl(corpus_path):
        title = str(rec.get("title", "")).strip()
        text = str(rec.get("text", "")).strip()
        if not title or not text:
            continue
        batch.append({"title": title, "text": text})
        
        if len(batch) >= int(args.batch_size):
            _flush(client, model, args.collection, batch)
            count += len(batch)
            print(f"已处理数据量: {count}")
            batch = []

    if batch:
        _flush(client, model, args.collection, batch)
        count += len(batch)
        print(f"处理完成，总计: {count}")

def _flush(client: QdrantClient, model: FlagAutoModel, collection: str, batch: list[dict]) -> None:
    texts = [f'{d["title"]} {d["text"]}' for d in batch]
    # 使用模型编码
    vectors = model.encode_corpus(texts)
    ids = [_stable_int_id(d["title"], d["text"]) for d in batch]
    
    # 写入 Docker 中的 Qdrant
    client.upsert(
        collection_name=collection,
        points=Batch(
            ids=ids,
            vectors=vectors.tolist(),
            payloads=batch,
        ),
    )

if __name__ == "__main__":
    main()