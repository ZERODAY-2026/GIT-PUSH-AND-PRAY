"""
metrics.py
==========
Metrics calculation for Stage 1: Retrieval Evaluation.

Computes:
- Recall@1, Recall@3, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- Silence Detection Accuracy (for negative/impossible clauses)
- Category-wise aggregate breakdown
"""
from __future__ import annotations

from typing import Any, Dict, List


def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate retrieval metrics from individual evaluation records.

    Each item in `results` should have:
    - is_positive: bool (True if gold clause was present, False if contract is silent)
    - gold_found: bool (True if gold was found within the retrieved chunks)
    - gold_rank: int | None (1-indexed rank of first relevant chunk, or None)
    - silence_detected: bool | None (for negative examples)
    - category: str
    """
    total_examples = len(results)
    if total_examples == 0:
        return {"error": "No results provided"}

    pos_examples = [r for r in results if r.get("is_positive", True)]
    neg_examples = [r for r in results if not r.get("is_positive", True)]

    # Positive Metrics (Recall@K and MRR)
    total_pos = len(pos_examples)
    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    r10_hits = 0
    reciprocal_ranks = []

    for r in pos_examples:
        rank = r.get("gold_rank")
        if rank is not None and rank > 0:
            reciprocal_ranks.append(1.0 / rank)
            if rank <= 1:
                r1_hits += 1
            if rank <= 3:
                r3_hits += 1
            if rank <= 5:
                r5_hits += 1
            if rank <= 10:
                r10_hits += 1
        else:
            reciprocal_ranks.append(0.0)

    recall_at_1 = (r1_hits / total_pos) if total_pos > 0 else 0.0
    recall_at_3 = (r3_hits / total_pos) if total_pos > 0 else 0.0
    recall_at_5 = (r5_hits / total_pos) if total_pos > 0 else 0.0
    recall_at_10 = (r10_hits / total_pos) if total_pos > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / total_pos) if total_pos > 0 else 0.0

    # Negative / Silence Metrics
    total_neg = len(neg_examples)
    silence_hits = sum(1 for r in neg_examples if r.get("silence_detected", False))
    silence_accuracy = (silence_hits / total_neg) if total_neg > 0 else 1.0

    # Category-wise Breakdown
    categories: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {
                "total_examples": 0,
                "positive_examples": 0,
                "silence_examples": 0,
                "r1_hits": 0,
                "r3_hits": 0,
                "r5_hits": 0,
                "r10_hits": 0,
                "rr_sum": 0.0,
                "silence_hits": 0,
            }
        c = categories[cat]
        c["total_examples"] += 1
        if r.get("is_positive", True):
            c["positive_examples"] += 1
            rank = r.get("gold_rank")
            if rank is not None and rank > 0:
                c["rr_sum"] += 1.0 / rank
                if rank <= 1:
                    c["r1_hits"] += 1
                if rank <= 3:
                    c["r3_hits"] += 1
                if rank <= 5:
                    c["r5_hits"] += 1
                if rank <= 10:
                    c["r10_hits"] += 1
        else:
            c["silence_examples"] += 1
            if r.get("silence_detected", False):
                c["silence_hits"] += 1

    category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, c in categories.items():
        p_count = c["positive_examples"]
        s_count = c["silence_examples"]
        category_summary[cat] = {
            "total_examples": c["total_examples"],
            "positive_examples": p_count,
            "silence_examples": s_count,
            "recall_at_1": round(c["r1_hits"] / p_count, 4) if p_count else 0.0,
            "recall_at_3": round(c["r3_hits"] / p_count, 4) if p_count else 0.0,
            "recall_at_5": round(c["r5_hits"] / p_count, 4) if p_count else 0.0,
            "recall_at_10": round(c["r10_hits"] / p_count, 4) if p_count else 0.0,
            "mrr": round(c["rr_sum"] / p_count, 4) if p_count else 0.0,
            "silence_accuracy": round(c["silence_hits"] / s_count, 4) if s_count else 1.0,
        }

    return {
        "total_examples_evaluated": total_examples,
        "positive_clause_examples": total_pos,
        "silence_negative_examples": total_neg,
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "silence_detection_accuracy": round(silence_accuracy, 4),
        "category_breakdown": category_summary,
    }
