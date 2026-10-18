
"""
test_chunker.py
===============
pytest test suite for Module 3 — Legal-aware Semantic Chunker.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.processing.chunker import (
    chunk_document,
    estimate_tokens,
    save_chunks,
    split_into_sentences,
)


class TestSentenceSplitter:
    def test_respects_legal_abbreviations(self):
        text = "Party A v. Party B was filed under Sec. 4. Tenant owes approx. $500. Next sentence starts here."
        sents = split_into_sentences(text)
        assert len(sents) == 3
        assert "Sec. 4" in sents[0]
        assert "approx. $500" in sents[1]
        assert "Next sentence starts here." in sents[2]


class TestChunkerProperties:
    @pytest.fixture
    def sample_cleaned_doc(self):
        return {
            "document_id": "lease_001",
            "source_file": "lease_001.pdf",
            "source_path": "/data/lease_001.pdf",
            "total_pages": 3,
            "pages": [
                {
                    "page_number": 1,
                    "text": (
                        "SECTION 1: PARTIES AND PREMISES\n\n"
                        "1.1 Landlord agrees to lease the premises to Tenant.\n\n"
                        "1.2 Term shall commence on January 1, 2026."
                    ),
                },
                {
                    "page_number": 2,
                    "text": (
                        "SECTION 2: RENT AND DEPOSIT\n\n"
                        "2.1 Rent shall be $2,000 per month payable in advance.\n\n"
                        "2.2 A security deposit of $4,000 is required upon signing."
                    ),
                },
                {
                    "page_number": 3,
                    "text": (
                        "SECTION 3: TERMINATION\n\n"
                        "3.1 Either party may terminate with 30 days prior written notice."
                    ),
                },
            ],
        }

    def test_chunks_are_non_empty(self, sample_cleaned_doc):
        chunks = chunk_document(sample_cleaned_doc, target_tokens=100)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c["text"].strip()) > 0
            assert c["char_count"] > 0
            assert c["token_count"] > 0

    def test_chunk_ids_are_unique_and_deterministic(self, sample_cleaned_doc):
        run1 = chunk_document(sample_cleaned_doc, target_tokens=100)
        run2 = chunk_document(sample_cleaned_doc, target_tokens=100)

        ids1 = [c["chunk_id"] for c in run1]
        ids2 = [c["chunk_id"] for c in run2]

        # Unique
        assert len(ids1) == len(set(ids1))
        # Deterministic
        assert ids1 == ids2

    def test_page_metadata_is_preserved(self, sample_cleaned_doc):
        chunks = chunk_document(sample_cleaned_doc, target_tokens=80)
        for c in chunks:
            assert "page_start" in c
            assert "page_end" in c
            assert c["page_start"] <= c["page_end"]
            assert 1 <= c["page_start"] <= 3

    def test_section_and_clause_preserved(self, sample_cleaned_doc):
        chunks = chunk_document(sample_cleaned_doc, target_tokens=60)
        # Verify detected sections
        sections = [c["section"] for c in chunks if c["section"] is not None]
        assert any("PARTIES" in s or "Section 1" in s for s in sections)
        assert any("RENT" in s or "Section 2" in s for s in sections)
        assert any("TERMINATION" in s or "Section 3" in s for s in sections)

    def test_does_not_invent_section_or_clause(self):
        doc = {
            "document_id": "plain_doc",
            "source_file": "plain.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "text": "This is a simple plain text document without any legal numbering or sections.",
                }
            ],
        }
        chunks = chunk_document(doc, target_tokens=100)
        assert len(chunks) == 1
        assert chunks[0]["section"] is None
        assert chunks[0]["clause"] is None

    def test_splits_oversized_unit_at_sentence_boundary(self):
        # Create a single huge paragraph
        long_para = " ".join([f"Sentence number {i} provides detailed legal obligations." for i in range(100)])
        doc = {
            "document_id": "big_doc",
            "source_file": "big.pdf",
            "pages": [{"page_number": 1, "text": long_para}],
        }
        chunks = chunk_document(doc, target_tokens=100, max_tokens=150)
        assert len(chunks) > 1
        for c in chunks:
            # Check sentence integrity: should not end with incomplete word
            assert c["text"].endswith(".")

    def test_save_chunks_to_disk(self, sample_cleaned_doc, tmp_path):
        chunks = chunk_document(sample_cleaned_doc)
        out_path = save_chunks(chunks, source_file="lease_001.pdf", output_dir=tmp_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(loaded) == len(chunks)
        assert loaded[0]["chunk_id"] == chunks[0]["chunk_id"]
