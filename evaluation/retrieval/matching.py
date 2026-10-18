"""
matching.py
===========
Evidence matching module for CUAD retrieval evaluation.

Determines whether a retrieved text chunk contains a sufficiently large portion
of the CUAD gold standard clause context.

Addresses real-world differences between gold annotations and indexed chunks:
1. Whitespace and line-break variations.
2. Punctuation and case differences.
3. Chunk boundary truncations (clause spanning across chunk boundaries).
4. Long clause annotations vs. targeted chunk extracts.
"""
from __future__ import annotations

import re
import string
from typing import List, Tuple, Union

# Common English stopwords to ignore during semantic overlap calculation
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Replace newlines, tabs, and dashes with spaces
    text = re.sub(r"[\r\n\t\-\_/]+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    return " ".join(text.split())


def get_content_tokens(text: str) -> List[str]:
    """Tokenize and return non-stopword alphanumeric tokens."""
    tokens = normalize_text(text).split()
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return content if content else tokens


def token_overlap_recall(retrieved: str, gold: str) -> float:
    """Calculate the fraction of gold content tokens present in the retrieved text."""
    gold_tokens = get_content_tokens(gold)
    if not gold_tokens:
        return 0.0

    retrieved_tokens_set = set(get_content_tokens(retrieved))
    gold_tokens_set = set(gold_tokens)

    intersection = gold_tokens_set & retrieved_tokens_set
    return len(intersection) / len(gold_tokens_set)


def is_chunk_relevant(
    chunk_text: str,
    gold_target: Union[str, List[str]],
    min_overlap: float = 0.50,
) -> Tuple[bool, float, str]:
    """
    Determine if a chunk matches gold evidence.

    Returns:
    --------
    (is_relevant, score, match_type)
        is_relevant : bool
        score : float between 0.0 and 1.0
        match_type : str describing the mechanism of match
    """
    if not chunk_text or not gold_target:
        return False, 0.0, "none"

    if isinstance(gold_target, str):
        gold_targets = [gold_target]
    else:
        gold_targets = gold_target

    best_score = 0.0
    best_type = "none"

    norm_chunk = normalize_text(chunk_text)

    for gold in gold_targets:
        if not gold or not gold.strip():
            continue

        norm_gold = normalize_text(gold)
        if not norm_gold:
            continue

        # 1. Exact Substring Match: Gold is wholly inside chunk
        if norm_gold in norm_chunk:
            return True, 1.0, "exact_substring"

        # 2. Chunk Substring Match: Chunk is wholly inside gold (boundary split)
        if len(norm_chunk) >= 50 and norm_chunk in norm_gold:
            score = min(1.0, len(norm_chunk) / max(len(norm_gold), 1))
            if score >= 0.3:
                return True, score, "chunk_in_gold"

        # 3. Significant Prefix / Suffix Match (first 40 chars of gold)
        gold_snippet = norm_gold[:min(40, len(norm_gold))]
        if len(gold_snippet) >= 20 and gold_snippet in norm_chunk:
            return True, 0.90, "prefix_match"

        # 4. Token Overlap Recall
        recall = token_overlap_recall(chunk_text, gold)
        if recall > best_score:
            best_score = recall
            best_type = f"token_overlap_{recall:.2f}"

        if recall >= min_overlap:
            return True, recall, best_type

    # Not reaching threshold
    is_hit = best_score >= min_overlap
    return is_hit, best_score, best_type
