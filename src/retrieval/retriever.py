"""Question retrieval over a single document's local FAISS index."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Embedder(Protocol):
    def embed_texts(self, texts: list[str], **kwargs: Any): ...


class SearchStore(Protocol):
    def search(self, query_vector, top_k: int = 5) -> list[dict[str, Any]]: ...


@dataclass
class RetrievalResult:
    question: str
    chunks: list[dict[str, Any]]
    status: str
    best_score: float | None
    message: str | None = None


class LegalRetriever:
    """Embed a question, retrieve top-k chunks, and reject weak evidence.

    `min_score` is a tunable cosine-similarity cutoff, not a probability or a
    calibrated confidence estimate. Evaluate it against representative queries.
    The store supplied here should contain only the document being queried.
    """
    def __init__(
        self,
        embedder: Embedder,
        store: SearchStore,
        *,
        top_k: int = 5,
        min_score: float = 0.50,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.query_prefix = query_prefix

    def retrieve(self, question: str) -> RetrievalResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        query_vector = self.embedder.embed_texts([self.query_prefix + question])
        candidates = self.store.search(query_vector, top_k=self.top_k)
        candidates = sorted(candidates, key=lambda item: float(item.get("score", -1)), reverse=True)
        if not candidates:
            return RetrievalResult(question, [], "no_results", None,
                                   "No passages were retrieved from this document.")

        best_score = float(candidates[0].get("score", -1.0))
        relevant = [item for item in candidates if float(item.get("score", -1.0)) >= self.min_score]
        if not relevant:
            return RetrievalResult(
                question, [], "low_relevance", best_score,
                "The retrieved passages may not address this question. The document does not provide enough clearly relevant evidence to answer reliably.",
            )
        return RetrievalResult(question, relevant, "ok", best_score)
