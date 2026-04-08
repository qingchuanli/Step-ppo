from your_module import WikiQdrantRetriever

retriever = WikiQdrantRetriever(
    qdrant_url="http://localhost:6333",
    collection_name="hpqa_corpus",
)

results = retriever.search("Who wrote Hamlet?", top_k=5)

print(f"num_results={len(results)}")
for i, p in enumerate(results, 1):
    print("=" * 80)
    print(f"rank={i}")
    print(f"id={p.pid}")
    print(f"title={p.title}")
    print(f"score={p.score:.4f}")
    print(f"text={p.text[:300]}")