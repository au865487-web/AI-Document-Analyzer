from app.retrieval.retriever import Retriever

r = Retriever()

queries = [
    "What is Abdullah's educational background?",
    "EDUCATION currently studying Grade 10 Sindh Board secondary education",
]

for query in queries:
    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    hits = r.search(query, top_k=15)

    for i, h in enumerate(hits, 1):
        print(
            f"{i}: chunk={h.chunk_index}, "
            f"page={h.page_number}, "
            f"score={h.score:.4f}"
        )
        print(h.text[:180].replace("\n", " "))
        print()