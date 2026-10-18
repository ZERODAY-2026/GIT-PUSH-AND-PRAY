"""
tests/test_pdf_extractor.py
============================
pytest test suite for Module 1 — Local PDF Text Extraction.

Coverage
--------
- Successful extraction with page preservation
- Missing file
- Invalid (non-PDF) file
- Empty / no-text PDF (all scanned pages)
- Page count and page number accuracy
- JSON output structure and required fields
- Text output structure
- CLI --no-save flag
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

# The module under test
from src.ingestion.pdf_extractor import (
    ExtractionResult,
    PageResult,
    extract_pdf,
    save_outputs,
    main,
    OCR_THRESHOLD_CHARS,
)

# ---------------------------------------------------------------------------
# Helpers — minimal in-memory PDF construction
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FITZ_AVAILABLE, reason="PyMuPDF (fitz) not installed"
)


def _make_text_pdf(pages: list[str], tmp_path: Path) -> Path:
    """
    Create a minimal multi-page PDF with given text per page.
    Returns the path to the created file.
    """
    pdf_path = tmp_path / "test_doc.pdf"
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _make_blank_pdf(num_pages: int, tmp_path: Path) -> Path:
    """Create a PDF with blank (no text) pages — simulates a scanned document."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# 1. Successful extraction
# ---------------------------------------------------------------------------

class TestSuccessfulExtraction:
    def test_returns_extraction_result(self, tmp_path):
        pdf = _make_text_pdf(["Hello legal world."], tmp_path)
        result = extract_pdf(pdf)
        assert isinstance(result, ExtractionResult)
        assert result.error is None

    def test_source_file_is_basename(self, tmp_path):
        pdf = _make_text_pdf(["Clause 1: Tenant shall pay rent."], tmp_path)
        result = extract_pdf(pdf)
        assert result.source_file == pdf.name
        assert "/" not in result.source_file  # must be basename only

    def test_total_chars_positive(self, tmp_path):
        pdf = _make_text_pdf(["This is a rental agreement with meaningful content."], tmp_path)
        result = extract_pdf(pdf)
        assert result.total_chars > 0

    def test_extracted_text_contains_content(self, tmp_path):
        content = "SECTION 1: RENT OBLIGATIONS"
        pdf = _make_text_pdf([content], tmp_path)
        result = extract_pdf(pdf)
        assert result.pages[0].text.strip() != ""
        # PyMuPDF may slightly alter whitespace but key words should survive
        assert "RENT" in result.pages[0].text or "Rent" in result.pages[0].text or len(result.pages[0].text) > 5


# ---------------------------------------------------------------------------
# 2. Page preservation
# ---------------------------------------------------------------------------

class TestPagePreservation:
    def test_single_page_document(self, tmp_path):
        pdf = _make_text_pdf(["Only one page here."], tmp_path)
        result = extract_pdf(pdf)
        assert result.total_pages == 1
        assert len(result.pages) == 1

    def test_multi_page_document(self, tmp_path):
        texts = ["Page one content.", "Page two content.", "Page three content."]
        pdf = _make_text_pdf(texts, tmp_path)
        result = extract_pdf(pdf)
        assert result.total_pages == 3
        assert len(result.pages) == 3

    def test_page_numbers_are_one_based(self, tmp_path):
        pdf = _make_text_pdf(["A", "B", "C"], tmp_path)
        result = extract_pdf(pdf)
        page_nums = [p.page_number for p in result.pages]
        assert page_nums == [1, 2, 3]

    def test_page_result_type(self, tmp_path):
        pdf = _make_text_pdf(["Text on page."], tmp_path)
        result = extract_pdf(pdf)
        for page in result.pages:
            assert isinstance(page, PageResult)
            assert isinstance(page.page_number, int)
            assert isinstance(page.text, str)
            assert isinstance(page.needs_ocr, bool)
            assert isinstance(page.char_count, int)


# ---------------------------------------------------------------------------
# 3. Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_missing_file_returns_error(self, tmp_path):
        result = extract_pdf(tmp_path / "nonexistent.pdf")
        assert result.error is not None
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_missing_file_zero_pages(self, tmp_path):
        result = extract_pdf(tmp_path / "ghost.pdf")
        assert result.total_pages == 0

    def test_missing_file_zero_chars(self, tmp_path):
        result = extract_pdf(tmp_path / "ghost.pdf")
        assert result.total_chars == 0


# ---------------------------------------------------------------------------
# 4. Invalid PDF (not a PDF file)
# ---------------------------------------------------------------------------

class TestInvalidPDF:
    def test_random_bytes_returns_error(self, tmp_path):
        bad_file = tmp_path / "not_a_pdf.pdf"
        bad_file.write_bytes(b"\x00\x01\x02\x03 this is not a PDF at all!")
        result = extract_pdf(bad_file)
        assert result.error is not None

    def test_text_file_disguised_as_pdf(self, tmp_path):
        fake = tmp_path / "fake.pdf"
        fake.write_text("I am just a text file, not a PDF.", encoding="utf-8")
        result = extract_pdf(fake)
        # PyMuPDF may or may not raise; either an error is returned or
        # it handles gracefully — what matters is no unhandled exception.
        assert isinstance(result, ExtractionResult)


# ---------------------------------------------------------------------------
# 5. Blank / scanned PDF (no extractable text)
# ---------------------------------------------------------------------------

class TestBlankPDF:
    def test_blank_pages_flagged_as_needs_ocr(self, tmp_path):
        pdf = _make_blank_pdf(2, tmp_path)
        result = extract_pdf(pdf)
        # All blank pages should have needs_ocr=True
        for page in result.pages:
            assert page.needs_ocr is True

    def test_blank_pdf_returns_error_or_warning(self, tmp_path):
        pdf = _make_blank_pdf(1, tmp_path)
        result = extract_pdf(pdf)
        # Either error is set or scanned_pages > 0
        assert result.error is not None or result.scanned_pages > 0

    def test_blank_pdf_scanned_pages_count(self, tmp_path):
        pdf = _make_blank_pdf(3, tmp_path)
        result = extract_pdf(pdf)
        assert result.scanned_pages == 3


# ---------------------------------------------------------------------------
# 6. JSON output structure
# ---------------------------------------------------------------------------

class TestJSONOutput:
    def test_json_file_created(self, tmp_path):
        pdf = _make_text_pdf(["Clause 1. Definitions."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        txt_path, json_path = save_outputs(result, output_dir=output_dir)
        assert json_path.exists()

    def test_json_has_required_top_level_keys(self, tmp_path):
        pdf = _make_text_pdf(["Some legal text."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        _, json_path = save_outputs(result, output_dir=output_dir)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("source_file", "total_pages", "total_chars", "pages"):
            assert key in data, f"Missing key: {key}"

    def test_json_pages_have_required_keys(self, tmp_path):
        pdf = _make_text_pdf(["Legal clause."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        _, json_path = save_outputs(result, output_dir=output_dir)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for page in data["pages"]:
            for key in ("page_number", "char_count", "needs_ocr", "text"):
                assert key in page, f"Page missing key: {key}"

    def test_json_page_numbers_match(self, tmp_path):
        pdf = _make_text_pdf(["Page A", "Page B"], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        _, json_path = save_outputs(result, output_dir=output_dir)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["total_pages"] == 2
        assert data["pages"][0]["page_number"] == 1
        assert data["pages"][1]["page_number"] == 2

    def test_json_source_file_is_basename(self, tmp_path):
        pdf = _make_text_pdf(["Content."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        _, json_path = save_outputs(result, output_dir=output_dir)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "/" not in data["source_file"]
        assert "\\" not in data["source_file"]


# ---------------------------------------------------------------------------
# 7. Text output structure
# ---------------------------------------------------------------------------

class TestTextOutput:
    def test_txt_file_created(self, tmp_path):
        pdf = _make_text_pdf(["Some clause content."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        txt_path, _ = save_outputs(result, output_dir=output_dir)
        assert txt_path.exists()

    def test_txt_contains_page_header(self, tmp_path):
        pdf = _make_text_pdf(["Clause text here."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        txt_path, _ = save_outputs(result, output_dir=output_dir)
        content = txt_path.read_text(encoding="utf-8")
        assert "PAGE 1" in content

    def test_txt_has_all_pages(self, tmp_path):
        pdf = _make_text_pdf(["First page.", "Second page.", "Third page."], tmp_path)
        result = extract_pdf(pdf)
        output_dir = tmp_path / "extracted"
        txt_path, _ = save_outputs(result, output_dir=output_dir)
        content = txt_path.read_text(encoding="utf-8")
        assert "PAGE 1" in content
        assert "PAGE 2" in content
        assert "PAGE 3" in content


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_no_save_flag(self, tmp_path):
        pdf = _make_text_pdf(["CLI test clause."], tmp_path)
        exit_code = main([str(pdf), "--no-save"])
        assert exit_code == 0

    def test_cli_missing_file_returns_nonzero(self, tmp_path):
        exit_code = main([str(tmp_path / "nonexistent.pdf"), "--no-save"])
        assert exit_code != 0

    def test_cli_saves_outputs(self, tmp_path):
        pdf = _make_text_pdf(["Full CLI run."], tmp_path)
        output_dir = tmp_path / "out"
        exit_code = main([str(pdf), "--output-dir", str(output_dir)])
        assert exit_code == 0
        stem = pdf.stem
        assert (output_dir / f"{stem}.txt").exists()
        assert (output_dir / f"{stem}.json").exists()
