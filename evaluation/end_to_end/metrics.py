"""
metrics.py
==========
Metrics and error taxonomy for Stage 3: End-to-End Pipeline Evaluation.

Classifies failure points across the pipeline:
- EXTRACTION FAILURE
- RETRIEVAL FAILURE
- LLM FAILURE
- GROUNDING FAILURE
- TECHNICAL FAILURE
- UNKNOWN / REQUIRES REVIEW
"""
from __future__ import annotations

from typing import Any, Dict, List


def classify_pipeline_result(
    is_positive: bool,
    retrieval_status: str,
    gold_found: bool,
    silence_detected: bool | None,
    f1_score: float,
    groundedness_score: float,
    unsupported_claim_rate: float,
    technical_error: bool = False,
) -> Dict[str, Any]:
    """
    Attribute system outcome to a specific stage in the pipeline.

    Returns:
    --------
    {
        "status": "SUCCESS" | "FAILURE",
        "failure_stage": str,
        "explanation": str
    }
    """
    if technical_error:
        return {
            "status": "FAILURE",
            "failure_stage": "TECHNICAL FAILURE",
            "explanation": "Exception occurred during pipeline execution (indexing, embedding, or generation)."
        }

    # Case A: Contract is silent on this category
    if not is_positive:
        if silence_detected:
            return {
                "status": "SUCCESS",
                "failure_stage": "NONE",
                "explanation": "Silence correctly identified; no spurious clauses or hallucinations retrieved."
            }
        else:
            return {
                "status": "FAILURE",
                "failure_stage": "RETRIEVAL FAILURE",
                "explanation": "False positive: Contract is silent, but system retrieved unrelated chunks with high confidence."
            }

    # Case B: Positive Clause (Clause exists in contract)
    if not gold_found:
        return {
            "status": "FAILURE",
            "failure_stage": "RETRIEVAL FAILURE",
            "explanation": "Retriever failed to surface the gold clause into the top chunks passed to Gemma."
        }

    # Gold was retrieved: check Gemma's answer
    if unsupported_claim_rate > 0.40:
        return {
            "status": "FAILURE",
            "failure_stage": "GROUNDING FAILURE",
            "explanation": "Gemma retrieved the clause but introduced extraneous unsupported statements."
        }

    if f1_score < 0.20 and groundedness_score < 0.30:
        return {
            "status": "FAILURE",
            "failure_stage": "LLM FAILURE",
            "explanation": "Gemma received the correct evidence chunk but failed to extract or express the operative terms."
        }

    return {
        "status": "SUCCESS",
        "failure_stage": "NONE",
        "explanation": "Both retrieval and LLM triage answered the query accurately and grounded in contract text."
    }


def compute_end_to_end_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate all end-to-end evaluation records into high-level metrics."""
    total = len(results)
    if total == 0:
        return {"error": "No end-to-end results"}

    pos_results = [r for r in results if r.get("is_positive", True)]
    neg_results = [r for r in results if not r.get("is_positive", True)]

    total_successes = sum(1 for r in results if r.get("evaluation", {}).get("status") == "SUCCESS")
    retrieval_successes = sum(1 for r in pos_results if r.get("retrieval", {}).get("gold_found", False))
    silence_successes = sum(1 for r in neg_results if r.get("retrieval", {}).get("silence_detected", False))

    failure_counts: Dict[str, int] = {
        "EXTRACTION FAILURE": 0,
        "RETRIEVAL FAILURE": 0,
        "LLM FAILURE": 0,
        "GROUNDING FAILURE": 0,
        "TECHNICAL FAILURE": 0,
        "UNKNOWN / REQUIRES REVIEW": 0,
    }

    for r in results:
        stage = r.get("evaluation", {}).get("failure_stage", "NONE")
        if stage in failure_counts:
            failure_counts[stage] += 1
        elif stage != "NONE":
            failure_counts["UNKNOWN / REQUIRES REVIEW"] += 1

    # Category breakdown
    cat_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "success": 0, "retrieval_hits": 0, "pos_count": 0}
        c = cat_stats[cat]
        c["total"] += 1
        if r.get("evaluation", {}).get("status") == "SUCCESS":
            c["success"] += 1
        if r.get("is_positive", True):
            c["pos_count"] += 1
            if r.get("retrieval", {}).get("gold_found", False):
                c["retrieval_hits"] += 1

    category_summary = {}
    for cat, c in cat_stats.items():
        category_summary[cat] = {
            "total_examples": c["total"],
            "success_rate": round(c["success"] / c["total"], 4) if c["total"] else 0.0,
            "retrieval_recall": round(c["retrieval_hits"] / c["pos_count"], 4) if c["pos_count"] else 1.0,
        }

    return {
        "total_examples_evaluated": total,
        "overall_end_to_end_accuracy": round(total_successes / total, 4),
        "positive_clause_recall": round(retrieval_successes / len(pos_results), 4) if pos_results else 0.0,
        "silence_accuracy": round(silence_successes / len(neg_results), 4) if neg_results else 1.0,
        "failure_stage_distribution": failure_counts,
        "category_breakdown": category_summary,
    }
