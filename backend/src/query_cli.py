"""Interactive CLI for querying one locally indexed legal document."""
from __future__ import annotations
import argparse
from src.answer_question import answer_question
from src.embedding.embedder import LocalEmbedder
from src.generation.gemma_client import LocalGemmaClient
from src.retrieval.retriever import LegalRetriever
from src.vector_store.faiss_store import FAISSVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about a local legal document index.")
    parser.add_argument("--store", default="data/vector_store", help="Directory containing index.faiss, mapping.json and chunks.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.50, help="Initial similarity cutoff; evaluate on your dataset before relying on it")
    parser.add_argument("--model", default=None, help="Model identifier as shown by the local LM Studio server")
    args = parser.parse_args()

    store = FAISSVectorStore.load(args.store)
    embedder = LocalEmbedder()
    if embedder.dimension != store.dimension:
        raise SystemExit(f"Embedding/index dimension mismatch: model={embedder.dimension}, index={store.dimension}")
    retriever = LegalRetriever(embedder, store, top_k=args.top_k, min_score=args.min_score)
    llm = LocalGemmaClient(model=args.model)
    print("Local legal document Q&A. Type 'exit' to quit. No remote LLM endpoint is accepted.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = answer_question(question, retriever, llm)
            print(f"\n{result['answer']}")
            if result.get("sources"):
                print("\nSources:")
                for source in result["sources"]:
                    pages = source.get("page_start")
                    if source.get("page_end") and source.get("page_end") != pages:
                        pages = f"{pages}-{source['page_end']}"
                    print(f"- {source.get('source_file')} | page(s) {pages} | section={source.get('section')} | clause={source.get('clause')} | score={source.get('score', 0):.3f}")
        except Exception as exc:
            # Deliberately avoid printing prompt/context or document text.
            print(f"Query failed: {type(exc).__name__}: {exc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
