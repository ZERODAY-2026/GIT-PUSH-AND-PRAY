"""Dense + BM25 + RRF + reranker retrieval, returning evidence only."""

from __future__ import annotations

from typing import Any

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import fuse_results
from src.retrieval.query_preprocessor import preprocess_query
from src.retrieval.reranker import LocalReranker


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        reranker: LocalReranker,
        *,
        initial_k: int = 30,
        final_k: int = 5,
        rrf_k: int = 60,
        min_rerank_score: float | None = None,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.min_rerank_score = min_rerank_score

    def retrieve(self, query: str, *, debug: bool = False) -> dict[str, Any]:
        normalized_query = preprocess_query(query)
        dense_results = self.dense.retrieve(normalized_query, self.initial_k)
        bm25_results = self.bm25.retrieve(normalized_query, self.initial_k)
        fused = fuse_results(
            dense_results,
            bm25_results,
            top_k=self.initial_k,
            rrf_k=self.rrf_k,
        )
        metadata = self.dense.store.chunks
        candidates = [
            {**metadata[item["chunk_id"]], **item}
            for item in fused
            if item["chunk_id"] in metadata
        ]
        reranked = self.reranker.rerank(
            normalized_query,
            candidates,
            top_k=self.final_k,
        )
        if self.min_rerank_score is not None:
            reranked = [
                item
                for item in reranked
                if item["rerank_score"] >= self.min_rerank_score
            ]
        result: dict[str, Any] = {
            "status": "ok" if reranked else "insufficient_retrieval_evidence",
            "query": normalized_query,
            "results": reranked,
            "counts": {
                "dense": len(dense_results),
                "bm25": len(bm25_results),
                "fused": len(fused),
                "reranked": len(reranked),
            },
        }
        if debug:
            result["debug"] = {
                "dense": dense_results,
                "bm25": bm25_results,
                "fused": fused,
            }
        return result
