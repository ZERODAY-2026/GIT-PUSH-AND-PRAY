"""Conservative query normalization for legal retrieval."""

from __future__ import annotations

import re


def preprocess_query(query: str) -> str:
    """Normalize whitespace without changing legal terms, numbers, or quotes."""
    normalized = re.sub(r"\s+", " ", query.strip())
    if not normalized:
        raise ValueError("Query cannot be empty")
    return normalized
