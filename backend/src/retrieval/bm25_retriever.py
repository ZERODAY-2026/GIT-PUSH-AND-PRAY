"""Persistent local BM25 retrieval over the canonical chunk metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)*|[A-Za-z]+(?:[-'][A-Za-z]+)*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Load a persisted BM25 corpus and resolve results by stable chunk ID."""

    def __init__(self, index_dir: str | Path):
        directory = Path(index_dir)
        corpus_path = directory / "corpus.json"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing BM25 corpus: {corpus_path}")
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.chunk_ids: list[str] = payload["chunk_ids"]
        self.texts: list[str] = payload["texts"]
        self.bm25 = BM25Okapi([tokenize(text) for text in self.texts])

    def retrieve(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[index],
                "bm25_score": float(scores[index]),
            }
            for index in ranked
        ]
