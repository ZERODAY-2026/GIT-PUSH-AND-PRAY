"""CLI for evidence-only hybrid retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embedding.embedder import LocalEmbedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LocalReranker
from src.vector_store.faiss_store import FAISSVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve legal evidence without generating an answer.")
    parser.add_argument("query")
    parser.add_argument("--index", default="data/vector_store/retrieval")
    parser.add_argument("--initial-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    index_dir = Path(args.index)
    metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    embedder = LocalEmbedder(model_name=args.model or metadata["embedding_model"])
    store = FAISSVectorStore.load(index_dir)
    retriever = HybridRetriever(
        DenseRetriever(store, embedder),
        BM25Retriever(index_dir / "bm25"),
        LocalReranker(args.reranker_model),
        initial_k=args.initial_k,
        final_k=args.top_k,
        rrf_k=args.rrf_k,
    )
    print(json.dumps(retriever.retrieve(args.query, debug=args.debug), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
