from qdrant_client import QdrantClient
from recipe.hotpotqa.utils import WikiQdrantRetriever

# 所有的执行代码都放进这个判断里
if __name__ == "__main__":
    URL = "http://172.17.0.1:6333"
    COL = "hpqa_corpus"

    c = QdrantClient(url=URL)
    print("exists:", c.collection_exists(COL))
    # 注意：如果容器还在索引，这里可能会稍微卡顿一下
    print("points_count:", c.get_collection(COL).points_count)

    r = WikiQdrantRetriever(
        qdrant_url=URL, 
        collection_name=COL, 
        embedding_model_name="BAAI/bge-large-en-v1.5"
    )

    print("--- Start Searching ---")
    res = r.search("Who wrote Hamlet?", top_k=3)
    
    print("hits:", len(res))
    for i, x in enumerate(res, 1):
        print(i, f"{x.score:.4f}", x.title)