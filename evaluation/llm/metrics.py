"""
metrics.py
==========
Metrics calculation for Stage 2: LLM Comprehension Evaluation.

Computes:
- Exact Match (EM)
- Token-level Precision, Recall, and F1
- Normalized Key-Term F1
- Groundedness Score (fraction of generated sentences supported by gold evidence)
- Unsupported Claim Rate (fraction of generated sentences containing outside information)
- Answer Relevance
"""
from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Set


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Check exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, ignoring markdown headers."""
    lines = text.split("\n")
    cleaned_lines = [l for l in lines if not l.strip().startswith("#") and len(l.strip()) > 10]
    full_text = " ".join(cleaned_lines)
    sentences = re.split(r"(?<=[.!?]) +", full_text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def evaluate_groundedness(prediction: str, gold_evidence: str, min_word_overlap: float = 0.40) -> Dict[str, Any]:
    """
    Evaluate whether sentences in prediction are supported by the gold evidence.

    Returns:
    --------
    {
        "groundedness_score": float (0.0 to 1.0),
        "unsupported_claim_rate": float (0.0 to 1.0),
        "total_claims": int,
        "grounded_claims": int,
        "unsupported_claims": int
    }
    """
    sentences = split_into_sentences(prediction)
    if not sentences:
        return {
            "groundedness_score": 1.0,
            "unsupported_claim_rate": 0.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "unsupported_claims": 0,
        }

    gold_words = set(normalize_answer(gold_evidence).split())

    grounded_count = 0
    unsupported_count = 0

    # Common disclaimer or triage phrases in Gemma responses that shouldn't count as ungrounded claims
    template_phrases = [
        "not legal advice", "preliminary document triage", "consult an attorney",
        "jurisdiction", "explicit document statements", "potential interpretations",
        "recommended next steps"
    ]

    for sent in sentences:
        norm_s = normalize_answer(sent)
        if any(tp in norm_s for tp in template_phrases):
            continue

        sent_words = [w for w in norm_s.split() if len(w) > 3]
        if not sent_words:
            continue

        overlap = sum(1 for w in sent_words if w in gold_words) / len(sent_words)
        if overlap >= min_word_overlap:
            grounded_count += 1
        else:
            unsupported_count += 1

    total = grounded_count + unsupported_count
    if total == 0:
        return {
            "groundedness_score": 1.0,
            "unsupported_claim_rate": 0.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "unsupported_claims": 0,
        }

    groundedness = grounded_count / total
    unsupported_rate = unsupported_count / total

    return {
        "groundedness_score": round(groundedness, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "total_claims": total,
        "grounded_claims": grounded_count,
        "unsupported_claims": unsupported_count,
    }


def compute_aggregate_llm_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual LLM test evaluations into summary metrics."""
    if not results:
        return {"error": "No LLM results to aggregate"}

    total = len(results)
    f1_list = [r.get("f1", 0.0) for r in results]
    recall_list = [r.get("recall", 0.0) for r in results]
    em_list = [r.get("exact_match", 0.0) for r in results]
    grounded_list = [r.get("groundedness_score", 1.0) for r in results]
    unsupported_list = [r.get("unsupported_claim_rate", 0.0) for r in results]

    avg_f1 = sum(f1_list) / total
    avg_recall = sum(recall_list) / total
    avg_em = sum(em_list) / total
    avg_groundedness = sum(grounded_list) / total
    avg_unsupported = sum(unsupported_list) / total

    # Category breakdown
    cat_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"count": 0, "f1_sum": 0.0, "grounded_sum": 0.0, "unsupported_sum": 0.0}
        c = cat_stats[cat]
        c["count"] += 1
        c["f1_sum"] += r.get("f1", 0.0)
        c["grounded_sum"] += r.get("groundedness_score", 1.0)
        c["unsupported_sum"] += r.get("unsupported_claim_rate", 0.0)

    category_summary = {}
    for cat, c in cat_stats.items():
        cnt = c["count"]
        category_summary[cat] = {
            "total_examples": cnt,
            "avg_f1": round(c["f1_sum"] / cnt, 4),
            "avg_groundedness": round(c["grounded_sum"] / cnt, 4),
            "avg_unsupported_rate": round(c["unsupported_sum"] / cnt, 4),
        }

    return {
        "total_examples_evaluated": total,
        "mean_token_f1": round(avg_f1, 4),
        "mean_key_term_recall": round(avg_recall, 4),
        "exact_match_rate": round(avg_em, 4),
        "mean_groundedness_score": round(avg_groundedness, 4),
        "mean_unsupported_claim_rate": round(avg_unsupported, 4),
        "category_breakdown": category_summary,
    }
