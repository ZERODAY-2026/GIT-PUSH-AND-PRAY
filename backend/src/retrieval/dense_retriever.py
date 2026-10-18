"""Dense retrieval over a BGE-M3-compatible local FAISS index."""

from __future__ import annotations

from typing import Any

from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore


class DenseRetriever:
    def __init__(self, store: FAISSVectorStore, embedder: LocalEmbedder):
        if store.dimension != embedder.dimension:
            raise ValueError(
                f"Embedding/index dimension mismatch: model={embedder.dimension}, index={store.dimension}"
            )
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = self.embedder.embed_texts([query])
        results = self.store.search(vector, top_k=top_k)
        return [
            {**result, "chunk_id": result["chunk_id"], "dense_score": result["score"]}
            for result in results
        ]
