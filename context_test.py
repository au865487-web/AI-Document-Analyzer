from app.retrieval.retriever import Retriever
from app.context.builder import build_context

r = Retriever()

hits = r.search("What is Abdullah's educational background?", top_k=8)
context = build_context(hits)

print("RETRIEVED:", len(hits))
print("CONTEXT SOURCES:", len(context.sources))
print("TOTAL CHARS:", context.total_characters)
print("TRUNCATED:", context.truncated)
print("\n--- CONTEXT ---\n")
print(context.formatted_context)