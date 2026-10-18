"""Local cross-encoder reranking for a small hybrid candidate pool."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)


class LocalReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_RERANKER_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
        scores = self._load_model().predict(pairs)
        ranked = []
        for candidate, score in zip(candidates, scores):
            ranked.append({**candidate, "rerank_score": float(score)})
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]
