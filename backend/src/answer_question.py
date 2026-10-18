"""Thin orchestration layer for local, document-grounded question answering."""
from __future__ import annotations
from typing import Any
from src.generation.prompt_builder import build_messages
from src.retrieval.retriever import LegalRetriever


def answer_question(question: str, retriever: LegalRetriever, llm: Any) -> dict[str, Any]:
    """Retrieve evidence and generate an answer, returning sources for the UI."""
    retrieval = retriever.retrieve(question)
    if retrieval.status != "ok":
        silence_answer = (
            "The document is silent on this matter. Based on semantic retrieval against the indexed document, "
            "no relevant clauses or provisions were found to answer this question reliably.\n\n"
            "### Silence & Risk Assessment\n"
            "- If this term or protection is customary or required by law in your jurisdiction, its absence in this document may represent a material omission or one-sided risk.\n"
            "- The document does not contain explicit provisions addressing this specific query.\n\n"
            "### Recommended Next Steps\n"
            "- Check whether separate addenda, schedules, or extrinsic agreements apply.\n"
            "- Consult a qualified legal professional licensed in your jurisdiction to assess statutory default rules.\n\n"
            "Not legal advice; preliminary document triage only."
        )
        return {
            "question": retrieval.question,
            "status": retrieval.status,
            "answer": silence_answer,
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
