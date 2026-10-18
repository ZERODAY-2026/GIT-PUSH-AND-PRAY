"""
test_cleaner.py
===============
pytest test suite for Module 2 — Legal Document Cleaner.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.processing.cleaner import (
    clean_document,
    clean_page_text,
    derive_document_id,
    detect_repeated_headers_footers,
    save_cleaned_document,
)


class TestCleanerPreservation:
    """Verify that critical legal terms, negations, numbers, and currency are preserved."""

    def test_preserves_negations_and_modals(self):
        text = (
            "The Tenant shall NOT sublet the premises without prior written consent. "
            "Neither party may assign this Agreement unless agreed in writing. "
            "Subletting is strictly prohibited."
        )
        cleaned = clean_page_text(text)
        assert "NOT" in cleaned
        assert "shall" in cleaned
        assert "without" in cleaned
        assert "Neither" in cleaned
        assert "unless" in cleaned
        assert "prohibited" in cleaned

    def test_preserves_numbers_and_monetary_amounts(self):
        text = (
            "The monthly rent is $2,500.00 (or ₹1,85,000) payable on the 1st day "
            "of each month, with a late fee of 5% if paid after 10 days, "
            "effective from 01 October 2026."
        )
        cleaned = clean_page_text(text)
        assert "$2,500.00" in cleaned
        assert "₹1,85,000" in cleaned
        assert "1st" in cleaned
        assert "5%" in cleaned
        assert "10 days" in cleaned
        assert "2026" in cleaned

    def test_preserves_clause_numbering_and_punctuation(self):
        text = (
            "SECTION 4 — RENT OBLIGATIONS\n\n"
            "4.1. Payment Schedule: Due on the first of each month.\n\n"
            "(a) Late penalty applies after 5 days.\n"
            "(b) Interest shall accrue at 1.5% per month."
        )
        cleaned = clean_page_text(text)
        assert "SECTION 4" in cleaned
        assert "4.1." in cleaned
        assert "(a)" in cleaned
        assert "(b)" in cleaned
        assert "1.5%" in cleaned


class TestWhitespaceAndLineWrap:
    """Verify whitespace normalization and safe line-wrap repair."""

    def test_normalizes_repeated_whitespace(self):
        text = "This   document    contains    excessive     spaces."
        cleaned = clean_page_text(text)
        assert cleaned == "This document contains excessive spaces."

    def test_normalizes_excessive_newlines(self):
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        cleaned = clean_page_text(text)
        assert cleaned == "Paragraph one.\n\nParagraph two."

    def test_repairs_unnecessary_line_breaks(self):
        text = (
            "The Tenant agrees to pay the Landlord the agreed\n"
            "monthly rent on the first day of each calendar\n"
            "month without demand or deduction."
        )
        cleaned = clean_page_text(text)
        assert (
            cleaned
            == "The Tenant agrees to pay the Landlord the agreed monthly rent on the first day of each calendar month without demand or deduction."
        )

    def test_does_not_join_clauses_or_headings(self):
        text = (
            "SECTION 7 — TERMINATION\n"
            "7.1 Either party may terminate with 30 days notice.\n"
            "7.2 The Tenant shall vacate the premises."
        )
        cleaned = clean_page_text(text)
        lines = cleaned.splitlines()
        # Should remain separate paragraphs/lines
        assert len(lines) >= 3


class TestHeaderFooterDetection:
    """Verify repeated boilerplate detection and removal."""

    def test_detects_and_removes_repeated_header(self):
        pages = [
            {
                "page_number": 1,
                "text": "CONFIDENTIAL RENTAL AGREEMENT\nPage 1 of 3\n\nActual clause text on page 1.\nFooter boilerplate",
            },
            {
                "page_number": 2,
                "text": "CONFIDENTIAL RENTAL AGREEMENT\nPage 2 of 3\n\nActual clause text on page 2.\nFooter boilerplate",
            },
            {
                "page_number": 3,
                "text": "CONFIDENTIAL RENTAL AGREEMENT\nPage 3 of 3\n\nActual clause text on page 3.\nFooter boilerplate",
            },
        ]
        doc = {
            "source_file": "lease.pdf",
            "pages": pages,
        }
        cleaned_doc = clean_document(doc, remove_headers_footers=True)

        for p in cleaned_doc["pages"]:
            # Header and footer boilerplate should be removed
            assert "CONFIDENTIAL RENTAL AGREEMENT" not in p["text"]
            assert "Footer boilerplate" not in p["text"]
            # Meaningful content must be preserved
            assert "Actual clause text" in p["text"]

    def test_preserves_content_when_not_repeated(self):
        pages = [
            {"page_number": 1, "text": "Unique top title\n\nPage 1 content."},
            {"page_number": 2, "text": "Different top title\n\nPage 2 content."},
        ]
        doc = {"source_file": "doc.pdf", "pages": pages}
        cleaned_doc = clean_document(doc, remove_headers_footers=True)
        # Unique headers should NOT be stripped
        assert "Unique top title" in cleaned_doc["pages"][0]["text"]
        assert "Different top title" in cleaned_doc["pages"][1]["text"]


class TestCleanDocumentEndToEnd:
    def test_structure_and_persistence(self, tmp_path):
        doc = {
            "source_file": "rental_agreement.pdf",
            "source_path": "/path/rental_agreement.pdf",
            "pages": [
                {"page_number": 1, "text": "Clause 1. Definitions and terms."},
                {"page_number": 2, "text": "Clause 2. Rent payment of $1,000."},
            ],
        }
        cleaned = clean_document(doc)
        assert cleaned["document_id"] == "rental_agreement"
        assert cleaned["total_pages"] == 2
        assert len(cleaned["pages"]) == 2
        assert cleaned["pages"][0]["page_number"] == 1
        assert cleaned["pages"][1]["page_number"] == 2

        out_path = save_cleaned_document(cleaned, output_dir=tmp_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["document_id"] == "rental_agreement"
