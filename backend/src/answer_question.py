"""Thin orchestration layer for local, document-grounded question answering."""
from __future__ import annotations
from typing import Any
from src.generation.prompt_builder import build_messages
from src.retrieval.retriever import LegalRetriever


def answer_question(question: str, retriever: LegalRetriever, llm: Any) -> dict[str, Any]:
    """Retrieve evidence and generate an answer, returning sources for the UI."""
    retrieval = retriever.retrieve(question)
    if retrieval.status != "ok":
        return {
            "question": retrieval.question,
            "status": retrieval.status,
            "answer": retrieval.message or "I could not find enough relevant evidence in this document to answer reliably.",
            "sources": [],
            "best_score": retrieval.best_score,
            "disclaimer": "Not legal advice; preliminary document triage only.",
        }
    messages = build_messages(retrieval.question, retrieval.chunks)
    answer = llm.generate(messages)
    return {
        "question": retrieval.question,
        "status": "ok",
        "answer": answer,
        "sources": [
            {k: chunk.get(k) for k in ("chunk_id", "source_file", "page_start", "page_end", "section", "clause", "score")}
            for chunk in retrieval.chunks
        ],
        "best_score": retrieval.best_score,
    }
