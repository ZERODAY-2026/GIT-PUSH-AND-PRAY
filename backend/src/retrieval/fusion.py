"""Rank-based fusion for hybrid legal retrieval."""

from __future__ import annotations

from typing import Any


def fuse_results(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    top_k: int = 30,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse dense and lexical rankings without comparing raw score scales."""
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k and rrf_k must be positive")

    fused: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        item = fused.setdefault(chunk_id, {"chunk_id": chunk_id})
        item["dense_rank"] = rank
        item["dense_score"] = result.get("dense_score", result.get("score"))
        item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        item = fused.setdefault(chunk_id, {"chunk_id": chunk_id})
        item["bm25_rank"] = rank
        item["bm25_score"] = result.get("bm25_score", result.get("score"))
        item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda item: float(item["rrf_score"]),
        reverse=True,
    )[:top_k]
