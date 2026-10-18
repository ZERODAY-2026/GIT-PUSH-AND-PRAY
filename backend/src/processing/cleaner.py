"""
cleaner.py
==========
Module 2 — Conservative, legal-aware text cleaning.

Responsibilities
----------------
- Accept extracted document data (from Module 1 JSON).
- Clean extracted page text while preserving legally meaningful information:
  * Preserve negations ("not", "never", "neither", "nor", etc.)
  * Preserve modal obligations ("shall", "must", "may", "will", "prohibited", etc.)
  * Preserve numbers, dates, percentages, and monetary amounts
  * Preserve clause numbers, section headings, and punctuation
  * Normalize excessive whitespace and repeated blank lines
  * Safely repair unnecessary mid-sentence line-wrap fragmentation
- Detect and remove repeated page headers and footers across pages with high confidence
- Save cleaned document structure to data/processed/<stem>_cleaned.json
- Perform text integrity verification to ensure no meaningful content is deleted
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("processing.cleaner")

# Threshold ratio below which we warn that excessive text may have been lost
INTEGRITY_WARNING_RATIO = 0.65

# Regex patterns for structural markers that should NOT be joined/unwrapped
RE_CLAUSE_START = re.compile(
    r"^(\d+(\.\d+)*|[A-Z]\.|\([a-z0-9ivx]+\)|SECTION\b|ARTICLE\b|CLAUSE\b|PART\b|WHEREAS\b|NOW THEREFORE\b|SCHEDULE\b|APPENDIX\b|[•\-\*])",
    re.IGNORECASE,
)
RE_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9\s,\-:;]{4,}$")
RE_PAGINATION_LINE = re.compile(
    r"^(page\s+\d+(\s+of\s+\d+)?|\d+(\s*/\s*\d+)?|\-\s*\d+\s*\-)$",
    re.IGNORECASE,
)


def derive_document_id(source_file: str) -> str:
    """Derive a clean, deterministic document identifier from a filename."""
    stem = Path(source_file).stem
    # Replace spaces and punctuation with underscores, lowercase
    clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return clean_id or "doc"


def _normalize_line_template(line: str) -> str:
    """
    Normalize a candidate header/footer line.
    Only mask page numbers when the line clearly represents pagination/boilerplate.
    """
    norm = line.strip().lower()
    # Mask variable page numbers e.g. "page 1 of 11", "page 1"
    if re.search(r"\bpage\s+\d+\b", norm) or RE_PAGINATION_LINE.match(line.strip()):
        norm = re.sub(r"\bpage\s+\d+\b", "page <num>", norm)
        norm = re.sub(r"\b\d+\s+of\s+\d+\b", "<num> of <num>", norm)
        norm = re.sub(r"^\d+$", "<num>", norm)
        norm = re.sub(r"^[-—–]\s*\d+\s*[-—–]$", "<num>", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _is_valid_header_footer_candidate(line: str) -> bool:
    """Check if a line can reasonably be a header or footer (not legal clauses)."""
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False
    # Never consider clause starts or section headers as boilerplate headers/footers
    if RE_CLAUSE_START.match(s):
        return False
    return True


def detect_repeated_headers_footers(
    pages: List[Dict[str, Any]], min_page_ratio: float = 0.5
) -> Tuple[Set[str], Set[str]]:
    """
    Detect repeated header and footer lines that occur consistently across pages.

    Only lines appearing in >= min_page_ratio (at least 2 pages) are considered.
    If uncertain, nothing is removed.
    """
    if len(pages) < 2:
        return set(), set()

    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}
    raw_header_map: Dict[str, str] = {}
    raw_footer_map: Dict[str, str] = {}

    for page in pages:
        text = page.get("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        # Check candidate headers (first 2 lines)
        for cand in lines[:2]:
            if _is_valid_header_footer_candidate(cand):
                template = _normalize_line_template(cand)
                header_counts[template] = header_counts.get(template, 0) + 1
                raw_header_map[template] = cand

        # Check candidate footer:
        # Check the last line; if it's pagination, also check the line right above it
        cand_footers = []
        if lines and _is_valid_header_footer_candidate(lines[-1]):
            cand_footers.append(lines[-1])
            if len(lines) > 2 and RE_PAGINATION_LINE.match(lines[-1]) and _is_valid_header_footer_candidate(lines[-2]):
                cand_footers.append(lines[-2])

        for cand in cand_footers:
            template = _normalize_line_template(cand)
            footer_counts[template] = footer_counts.get(template, 0) + 1
            raw_footer_map[template] = cand

    min_occurrences = max(2, int(len(pages) * min_page_ratio))

    confirmed_headers = {
        template
        for template, count in header_counts.items()
        if count >= min_occurrences
    }
    confirmed_footers = {
        template
        for template, count in footer_counts.items()
        if count >= min_occurrences
    }

    for h in confirmed_headers:
        logger.info(
            "Detected repeated header boilerplate (%d occurrences): '%s'",
            header_counts[h],
            raw_header_map.get(h, h),
        )

    for f in confirmed_footers:
        logger.info(
            "Detected repeated footer boilerplate (%d occurrences): '%s'",
            footer_counts[f],
            raw_footer_map.get(f, f),
        )

    return confirmed_headers, confirmed_footers


def clean_page_text(
    text: str,
    header_templates: Optional[Set[str]] = None,
    footer_templates: Optional[Set[str]] = None,
) -> str:
    """
    Clean text for a single page with conservative, legal-safe rules.

    Rules:
    - Preserves all letters, numbers, punctuation, negations, obligations.
    - Strips confirmed repeated headers and footers.
    - Normalizes horizontal spacing within lines.
    - Repairs mid-sentence line-wrap fragmentation where safe.
    - Preserves empty line paragraph breaks and structural clause headings.
    """
    if not text or not text.strip():
        return ""

    header_templates = header_templates or set()
    footer_templates = footer_templates or set()

    raw_lines = text.splitlines()
    non_empty_indices = [i for i, line in enumerate(raw_lines) if line.strip()]

    if not non_empty_indices:
        return ""

    # Identify lines to remove if they match confirmed headers (near top) or footers (near bottom)
    lines_to_skip = set()
    first_few = non_empty_indices[:2]
    # Ensure last_few does not overlap with first_few
    last_few = [idx for idx in non_empty_indices[-2:] if idx not in first_few]

    for idx in first_few:
        line_clean = raw_lines[idx].strip()
        tpl = _normalize_line_template(line_clean)
        if tpl in header_templates or RE_PAGINATION_LINE.match(line_clean):
            lines_to_skip.add(idx)

    for idx in last_few:
        line_clean = raw_lines[idx].strip()
        tpl = _normalize_line_template(line_clean)
        if tpl in footer_templates or RE_PAGINATION_LINE.match(line_clean):
            lines_to_skip.add(idx)

    # Filter out skipped lines while preserving paragraph spacing structure
    filtered_lines: List[str] = []
    for i, line in enumerate(raw_lines):
        if i in lines_to_skip:
            continue
        filtered_lines.append(line)

    # Normalize horizontal whitespace on each line
    norm_lines: List[str] = []
    for line in filtered_lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        norm_lines.append(line)

    # Smart line-wrap repair:
    # Join line i to line i+1 if line i does not end with terminal punctuation
    # and line i+1 is not a heading, clause start, or blank line.
    paragraphs: List[str] = []
    current_block: List[str] = []

    for line in norm_lines:
        if not line:
            if current_block:
                paragraphs.append(" ".join(current_block))
                current_block = []
            continue

        # If line looks like a structural clause or heading, flush current block
        is_structure = bool(RE_CLAUSE_START.match(line)) or (
            bool(RE_ALL_CAPS_HEADING.match(line)) and len(line) < 80
        )

        if is_structure and current_block:
            paragraphs.append(" ".join(current_block))
            current_block = [line]
            continue

        if not current_block:
            current_block.append(line)
        else:
            prev = current_block[-1]
            # Check if previous line ended with terminal punctuation or colon/semicolon
            ends_with_terminal = bool(re.search(r"[\.:;?!—–\-]\s*$", prev))
            if ends_with_terminal:
                # Keep separate line
                paragraphs.append(" ".join(current_block))
                current_block = [line]
            else:
                # Safe to un-wrap mid-sentence break
                current_block.append(line)

    if current_block:
        paragraphs.append(" ".join(current_block))

    # Join paragraphs with double newlines
    cleaned_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    # Ensure excessive blank lines are normalized
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text


def clean_document(
    extracted_data: Dict[str, Any],
    remove_headers_footers: bool = True,
) -> Dict[str, Any]:
    """
    Clean an entire extracted document structure.

    Parameters
    ----------
    extracted_data : dict
        Parsed JSON from Module 1 (containing source_file, pages, etc.).
    remove_headers_footers : bool, optional
        Whether to detect and remove repeated boilerplate headers/footers (default True).

    Returns
    -------
    dict
        Cleaned document structure with document_id and cleaned pages.
    """
    source_file = extracted_data.get("source_file", "document.pdf")
    source_path = extracted_data.get("source_path", "")
    pages = extracted_data.get("pages", [])
    document_id = derive_document_id(source_file)

    total_chars_original = sum(len(p.get("text", "")) for p in pages)

    header_templates: Set[str] = set()
    footer_templates: Set[str] = set()

    if remove_headers_footers and len(pages) > 1:
        header_templates, footer_templates = detect_repeated_headers_footers(pages)

    cleaned_pages: List[Dict[str, Any]] = []
    total_chars_cleaned = 0

    for page in pages:
        p_num = page.get("page_number", 1)
        raw_text = page.get("text", "")
        cleaned_text = clean_page_text(
            raw_text,
            header_templates=header_templates,
            footer_templates=footer_templates,
        )
        cleaned_chars = len(cleaned_text)
        total_chars_cleaned += cleaned_chars

        cleaned_pages.append(
            {
                "page_number": p_num,
                "char_count": cleaned_chars,
                "text": cleaned_text,
            }
        )

    # Integrity verification
    if total_chars_original > 100:
        ratio = total_chars_cleaned / total_chars_original
        if ratio < INTEGRITY_WARNING_RATIO:
            logger.warning(
                "Integrity warning for '%s': cleaned text retained only %.1f%% "
                "of original characters (%d -> %d chars). Inspect cleaning output.",
                source_file,
                ratio * 100,
                total_chars_original,
                total_chars_cleaned,
            )
        else:
            logger.info(
                "Document '%s' cleaned: %d pages, %d -> %d chars (%.1f%% retained)",
                source_file,
                len(cleaned_pages),
                total_chars_original,
                total_chars_cleaned,
                ratio * 100,
            )

    return {
        "document_id": document_id,
        "source_file": source_file,
        "source_path": source_path,
        "total_pages": len(cleaned_pages),
        "total_chars_original": total_chars_original,
        "total_chars_cleaned": total_chars_cleaned,
        "pages": cleaned_pages,
    }


def save_cleaned_document(
    cleaned_data: Dict[str, Any],
    output_dir: str | Path = "data/processed",
) -> Path:
    """Save cleaned document structure to JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(cleaned_data.get("source_file", "document")).stem
    out_path = out_dir / f"{stem}_cleaned.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    logger.info("Cleaned document saved to: %s", out_path)
    return out_path
