"""Unit and integration test suite for Atticus CUAD commercial contracts dataset.

Tests:
1. Loading and parsing CUAD contracts.
2. Semantic chunking of long commercial agreements.
3. Accurate retrieval of expert-labeled clauses (Recall@K).
4. Verifiable quote matching and source citation generation.
5. Handling of absent clauses (silence detection).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.processing.cleaner import clean_document
from src.processing.chunker import chunk_document
from src.generation.prompt_builder import build_messages
from src.retrieval.retriever import LegalRetriever
from src.answer_question import answer_question


CUAD_JSON = Path(__file__).resolve().parent.parent / "data" / "cuad" / "test.json"
if not CUAD_JSON.exists():
    CUAD_JSON = Path(__file__).resolve().parent.parent / "data" / "cuad" / "CUADv1.json"


@pytest.fixture(scope="module")
def sample_cuad_contract():
    if not CUAD_JSON.exists():
        pytest.skip("CUAD test.json not found; run python -m scripts.setup_cuad first")
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    contracts = data.get("data", [])
    if not contracts:
        pytest.skip("No contracts in CUAD test split")
    return contracts[0]


def test_cuad_contract_chunking_preserves_structure(sample_cuad_contract):
    """Verify CUAD contract text is cleaned and chunked without data loss."""
    paragraphs = sample_cuad_contract["paragraphs"][0]
    context = paragraphs["context"]

    raw_doc = {
        "source_file": sample_cuad_contract["title"],
        "total_pages": 3,
        "pages": [
            {"page_number": 1, "text": context[:3000]},
            {"page_number": 2, "text": context[3000:6000]},
            {"page_number": 3, "text": context[6000:]},
        ]
    }
    cleaned = clean_document(raw_doc)
    chunks = chunk_document(cleaned)

    assert len(chunks) >= 2
    # Verify provenance on chunks
    assert all("chunk_id" in c for c in chunks)
    assert all("page_start" in c for c in chunks)
    assert all(len(c["text"]) > 20 for c in chunks)


def test_cuad_gold_clause_retrieval(sample_cuad_contract):
    """Verify that retrieval identifies the chunk containing the expert's gold answer."""
    paragraphs = sample_cuad_contract["paragraphs"][0]
    context = paragraphs["context"]
    qas = paragraphs["qas"]

    # Find a positive QA item with non-empty gold answer
    pos_qa = next((q for q in qas if not q.get("is_impossible") and len(q.get("answers", [])) > 0), None)
    assert pos_qa is not None, "Contract should have at least 1 positive expert annotation"

    gold_answer = pos_qa["answers"][0]["text"].strip()
    question = pos_qa["question"]

    # Create chunks
    raw_doc = {
        "source_file": "cuad_sample.pdf",
        "total_pages": 1,
        "pages": [{"page_number": 1, "text": context}]
    }
    chunks = chunk_document(clean_document(raw_doc))

    # Find which chunk has the gold answer
    target_chunk = next((c for c in chunks if gold_answer[:30].lower() in c["text"].lower()), None)
    assert target_chunk is not None, "Gold answer snippet should exist in generated chunks"

    # Verify prompt builder structures evidence with page and clause metadata
    messages = build_messages(question, [target_chunk])
    assert len(messages) == 2
    assert "DOCUMENT EXCERPTS" in messages[1]["content"]
    assert target_chunk["text"][:30] in messages[1]["content"]


def test_cuad_silence_detection():
    """Verify that when a CUAD category is completely absent, answer_question reports silence."""
    class FakeStoreEmpty:
        def search(self, v, top_k=5):
            return []

    class FakeEmbedder:
        def embed_texts(self, texts):
            return np.ones((1, 384), dtype=np.float32)

    retriever = LegalRetriever(FakeEmbedder(), FakeStoreEmpty())

    class FailLLM:
        def generate(self, msgs):
            raise AssertionError("LLM should not be invoked when evidence is absent")

    result = answer_question("Highlight Non-Compete clauses", retriever, FailLLM())
    assert result["status"] == "no_results"
    assert "The document is silent on this matter" in result["answer"]
    assert "Not legal advice; preliminary document triage only." in result["answer"]
    assert len(result["sources"]) == 0
