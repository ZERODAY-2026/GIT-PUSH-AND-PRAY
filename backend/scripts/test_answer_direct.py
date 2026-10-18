import time
from src.answer_question import answer_question
from src.retrieval.retriever import LegalRetriever
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.generation.gemma_client import LocalGemmaClient
from pathlib import Path

print("Loading local vector store and embedder...", flush=True)
t0 = time.time()
store = FAISSVectorStore.load(Path("data/vector_store"))
embedder = LocalEmbedder()
retriever = LegalRetriever(embedder, store, top_k=3, min_score=0.45)
llm = LocalGemmaClient(max_tokens=1200)

print(f"Components loaded in {time.time()-t0:.2f}s. Sending question to Gemma 4 E4B...", flush=True)
t1 = time.time()
result = answer_question("What are the termination conditions and notice periods?", retriever, llm)
elapsed = time.time() - t1

print(f"\nQuery completed in {elapsed:.2f}s!", flush=True)
print("Status:", result["status"])
print("Best score:", result["best_score"])
print(f"Sources ({len(result['sources'])}):")
for s in result["sources"]:
    print(f" - Page {s['page_start']} | Clause {s['clause']} | Score: {s['score']:.3f}")

print("\n" + "=" * 65)
print("ANSWER FROM LOCAL GEMMA 4 E4B:")
print("=" * 65)
print(result["answer"])
print("=" * 65)
