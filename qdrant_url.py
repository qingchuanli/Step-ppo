from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

LOCAL_DB = "/root/data/qdrant_db"              # 你的本地 qdrant_db 路径
REMOTE_URL = "http://127.0.0.1:6333"           # 你的服务端 URL
COL = "hpqa_corpus"
BATCH = 2000

src = QdrantClient(path=LOCAL_DB)
dst = QdrantClient(url=REMOTE_URL)

if not src.collection_exists(COL):
    raise RuntimeError(f"source collection not found: {COL}")

# 从源集合读取向量维度
src_info = src.get_collection(COL)
vsize = src_info.config.params.vectors.size

# 目标端重建同名集合（如不想覆盖，把 recreate_collection 改为 create_collection + 判断）
dst.recreate_collection(
    collection_name=COL,
    vectors_config=VectorParams(size=vsize, distance=Distance.COSINE),
)

# 可选：给 title 建索引
try:
    dst.create_payload_index(COL, "title", field_schema="keyword")
except Exception:
    pass

offset = None
total = 0

while True:
    points, offset = src.scroll(
        collection_name=COL,
        limit=BATCH,
        with_payload=True,
        with_vectors=True,
        offset=offset,
    )
    if not points:
        break

    to_upsert = [
        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
        for p in points
    ]
    dst.upsert(collection_name=COL, points=to_upsert, wait=True)
    total += len(to_upsert)
    print(f"migrated: {total}")

    if offset is None:
        break

print("done, total:", total)
PY