"""
pdf_extractor.py
================
Module 1 — Local PDF Text Extraction for the localize-legal-RAG pipeline.

Responsibilities
----------------
- Accept a local PDF path via CLI or as a Python import.
- Extract text with page-level structure using PyMuPDF (fitz).
- Detect scanned / image-only PDFs and delegate to a future local OCR stub.
- Write structured JSON and plain-text outputs to data/extracted/.
- Never make network requests; never log document content; never call
  any external API.  Privacy is a hard architectural constraint.

Future pipeline position
------------------------
PDF → [THIS MODULE] → Cleaning → Chunking → Embeddings → FAISS →
Gemma → Grounded explanation + Citation + Safety warning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Third-party — PyMuPDF (imported as fitz)
# ---------------------------------------------------------------------------
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # fallback
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyMuPDF is required.  Install it with:  pip install pymupdf"
        ) from exc


# ---------------------------------------------------------------------------
# Logging — log only metadata, never document content
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pdf_extractor")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# A page is considered "text-bearing" if it has at least this many characters.
# Below this threshold the page is flagged as likely scanned / image-only.
OCR_THRESHOLD_CHARS: int = 30

# Default output directory relative to the project root (resolved at runtime).
DEFAULT_OUTPUT_DIR: Path = Path("data") / "extracted"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class PageResult:
    """Structured representation of a single extracted PDF page."""

    page_number: int          # 1-based
    char_count: int           # character count of extracted text
    needs_ocr: bool           # True -> page is likely scanned / image-only
    text: str                 # raw extracted text (may be empty if needs_ocr)
    blocks: List[dict] = field(default_factory=list)
    # blocks: list of PyMuPDF block dicts kept for downstream consumers
    # (chunker, layout analyser, etc.).  Each block contains:
    #   {"bbox": [...], "type": int, "text": str}


@dataclass
class ExtractionResult:
    """Top-level result returned by extract_pdf()."""

    source_file: str          # basename of the input file
    source_path: str          # absolute path (never sent anywhere)
    total_pages: int
    total_chars: int
    scanned_pages: int        # pages that triggered the OCR threshold
    pages: List[PageResult] = field(default_factory=list)
    error: Optional[str] = None   # non-None means extraction failed


# ---------------------------------------------------------------------------
# OCR stub — local OCR integration point (future)
# ---------------------------------------------------------------------------
def _ocr_page_stub(page: fitz.Page) -> str:  # type: ignore[name-defined]
    """
    Placeholder for future LOCAL OCR processing (e.g., via Tesseract / pytesseract).

    This function is called when a page contains fewer than OCR_THRESHOLD_CHARS
    characters of native text, indicating it is likely a scanned image.

    To integrate Tesseract in a future iteration:
        1.  pip install pytesseract Pillow
        2.  Install Tesseract binary:  https://github.com/UB-Mannheim/tesseract/wiki
        3.  Replace the body of this function with:

            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img)

    IMPORTANT: OCR must remain LOCAL.  Do NOT call any cloud OCR endpoint here.
    """
    # For now we return an empty string and let the caller handle the gap.
    return ""


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------
def _extract_page(page: fitz.Page, page_number: int) -> PageResult:  # type: ignore[name-defined]
    """
    Extract text and block structure from a single PyMuPDF page object.

    Block types returned by get_text("dict"):
        0 -> text block
        1 -> image block (skipped for text extraction)

    The block-level information is retained so downstream chunking code can
    reconstruct paragraph / heading boundaries without re-parsing raw text.
    """
    raw_dict = page.get_text("dict")  # richest structured representation
    text_parts: List[str] = []
    structured_blocks: List[dict] = []

    for block in raw_dict.get("blocks", []):
        block_type = block.get("type", -1)

        if block_type == 0:  # text block
            lines: List[str] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans)
                if line_text.strip():
                    lines.append(line_text)

            block_text = "\n".join(lines)

            if block_text.strip():
                text_parts.append(block_text)
                structured_blocks.append(
                    {
                        "bbox": block.get("bbox"),
                        "type": block_type,
                        "text": block_text,
                    }
                )
        # Image blocks (type 1) are intentionally skipped for now.
        # Future: pass image blocks to _ocr_page_stub.

    extracted_text = "\n\n".join(text_parts)
    char_count = len(extracted_text.strip())
    needs_ocr = char_count < OCR_THRESHOLD_CHARS

    if needs_ocr:
        logger.info(
            "Page %d has only %d chars — flagged for local OCR (stub called).",
            page_number,
            char_count,
        )
        ocr_text = _ocr_page_stub(page)
        if ocr_text.strip():
            extracted_text = ocr_text
            char_count = len(extracted_text.strip())

    return PageResult(
        page_number=page_number,
        char_count=char_count,
        needs_ocr=needs_ocr,
        text=extracted_text,
        blocks=structured_blocks,
    )


def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    """
    Main entry point for PDF extraction.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the input PDF.

    Returns
    -------
    ExtractionResult
        Structured extraction result.  Check result.error before use.

    Privacy guarantee
    -----------------
    This function:
      - reads only the local file at pdf_path
      - makes no network requests
      - logs only metadata (filename, page count, char count)
      - does not log document content
    """
    pdf_path = Path(pdf_path).resolve()
    basename = pdf_path.name

    logger.info("Processing: %s", basename)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not pdf_path.exists():
        msg = f"File not found: {pdf_path}"
        logger.error(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )

    if not pdf_path.is_file():
        msg = f"Path is not a file: {pdf_path}"
        logger.error(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )

    if pdf_path.suffix.lower() != ".pdf":
        msg = f"File does not have a .pdf extension: {basename}"
        logger.warning(msg)
        # Still attempt extraction — PyMuPDF will raise if truly invalid.

    # ------------------------------------------------------------------
    # Open PDF
    # ------------------------------------------------------------------
    try:
        doc: fitz.Document = fitz.open(str(pdf_path))  # type: ignore[attr-defined]
    except fitz.FileDataError as exc:
        msg = f"Invalid or malformed PDF — PyMuPDF could not open it: {exc}"
        logger.error(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error opening PDF: {exc}"
        logger.error(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )

    # ------------------------------------------------------------------
    # Password protection
    # ------------------------------------------------------------------
    if doc.needs_pass:
        doc.close()
        msg = (
            "PDF is password-protected.  "
            "Provide an unencrypted version or implement decryption locally."
        )
        logger.error(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )

    # ------------------------------------------------------------------
    # Empty PDF guard
    # ------------------------------------------------------------------
    if doc.page_count == 0:
        doc.close()
        msg = "PDF contains zero pages."
        logger.warning(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=0,
            total_chars=0,
            scanned_pages=0,
            error=msg,
        )

    # ------------------------------------------------------------------
    # Extract page by page
    # ------------------------------------------------------------------
    pages: List[PageResult] = []
    total_chars = 0
    scanned_pages = 0

    logger.info("Total pages: %d", doc.page_count)

    for page_index in range(doc.page_count):
        page_obj: fitz.Page = doc[page_index]  # type: ignore[index]
        page_result = _extract_page(page_obj, page_number=page_index + 1)
        pages.append(page_result)
        total_chars += page_result.char_count
        if page_result.needs_ocr:
            scanned_pages += 1

    doc.close()

    # ------------------------------------------------------------------
    # No-text-at-all guard (all pages were blank / scanned)
    # ------------------------------------------------------------------
    if total_chars == 0:
        msg = (
            "No extractable text found in any page.  "
            "The PDF may be entirely image-based and will require local OCR."
        )
        logger.warning(msg)
        return ExtractionResult(
            source_file=basename,
            source_path=str(pdf_path),
            total_pages=len(pages),
            total_chars=0,
            scanned_pages=scanned_pages,
            pages=pages,
            error=msg,
        )

    logger.info(
        "Extraction complete — pages: %d, chars: %d, scanned/OCR-needed: %d",
        len(pages),
        total_chars,
        scanned_pages,
    )

    return ExtractionResult(
        source_file=basename,
        source_path=str(pdf_path),
        total_pages=len(pages),
        total_chars=total_chars,
        scanned_pages=scanned_pages,
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------
def _build_json_payload(result: ExtractionResult) -> dict:
    """
    Build the JSON-serialisable dict that is written to disk.

    Blocks are excluded from the JSON output (they are large and internal).
    They remain available on PageResult objects for in-process consumers.
    """
    return {
        "source_file": result.source_file,
        "source_path": result.source_path,
        "total_pages": result.total_pages,
        "total_chars": result.total_chars,
        "scanned_pages": result.scanned_pages,
        "extraction_error": result.error,
        "pages": [
            {
                "page_number": p.page_number,
                "char_count": p.char_count,
                "needs_ocr": p.needs_ocr,
                "text": p.text,
            }
            for p in result.pages
        ],
    }


def save_outputs(
    result: ExtractionResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """
    Persist the extraction result to disk.

    Writes:
      <output_dir>/<stem>.txt  — plain text, one page block per line
      <output_dir>/<stem>.json — structured JSON with page metadata

    Returns
    -------
    (txt_path, json_path)

    Raises
    ------
    RuntimeError if result.error is set (nothing meaningful to save).
    """
    if result.error and not result.pages:
        raise RuntimeError(
            f"Cannot save outputs — extraction failed: {result.error}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(result.source_file).stem

    # --- plain text ---
    txt_path = output_dir / f"{stem}.txt"
    with txt_path.open("w", encoding="utf-8") as fh:
        for page in result.pages:
            fh.write(f"{'=' * 60}\n")
            fh.write(f"PAGE {page.page_number}\n")
            fh.write(f"{'=' * 60}\n\n")
            fh.write(page.text)
            fh.write("\n\n")

    logger.info("Text output written: %s", txt_path.name)

    # --- structured JSON ---
    json_path = output_dir / f"{stem}.json"
    payload = _build_json_payload(result)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("JSON output written: %s", json_path.name)

    return txt_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_extractor",
        description=(
            "Module 1 — Local PDF text extraction for the localize-legal-RAG pipeline.\n"
            "All processing is performed on-device.  No data leaves this machine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        metavar="PDF_PATH",
        help="Path to the input PDF file, e.g. data/input/rental_agreement.pdf",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Perform extraction but skip writing output files (useful for testing).",
    )
    return parser


def _print_summary(
    result: ExtractionResult,
    txt_path: Optional[Path],
    json_path: Optional[Path],
) -> None:
    """Print a concise processing summary.  Never prints document content."""
    print()
    print("-" * 50)
    print("  localize-legal-RAG  *  PDF Extraction Summary")
    print("-" * 50)
    print(f"  PDF             : {result.source_file}")
    if result.error:
        print(f"  WARNING         : {result.error}")
    print(f"  Pages           : {result.total_pages}")
    print(f"  Extracted chars : {result.total_chars:,}")
    if result.scanned_pages:
        print(
            f"  Scanned pages   : {result.scanned_pages} "
            "(OCR stub called — integrate local Tesseract for full support)"
        )
    if txt_path:
        print(f"  Output (text)   : {txt_path}")
    if json_path:
        print(f"  Output (JSON)   : {json_path}")
    print("-" * 50)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = extract_pdf(args.pdf_path)

    txt_path: Optional[Path] = None
    json_path: Optional[Path] = None

    if not args.no_save:
        try:
            txt_path, json_path = save_outputs(result, output_dir=args.output_dir)
        except RuntimeError as exc:
            logger.error("Could not save outputs: %s", exc)
            _print_summary(result, None, None)
            return 1

    _print_summary(result, txt_path, json_path)

    # Exit 1 if there was a hard error (file not found, invalid PDF, etc.)
    return 1 if (result.error and result.total_chars == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
