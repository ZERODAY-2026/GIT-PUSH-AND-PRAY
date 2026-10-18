"""
chunker.py
==========
Module 3 — Legal-aware semantic chunking.

Responsibilities
----------------
- Split cleaned legal document text into structured, coherent chunks.
- Prioritize semantic legal boundaries:
  Document -> Section -> Clause -> Sub-clause -> Paragraph -> Sentence fallback.
- Avoid splitting in the middle of a legal clause or sentence wherever practical.
- Generate deterministic, unique, reproducible chunk IDs:
  e.g. {document_id}_p{page:03d}_c{seq:03d}
- Maintain complete provenance:
  * chunk_id
  * document_id
  * source_file
  * page_start & page_end
  * section (or None if not detected; never invented)
  * clause (or None if not detected; never invented)
  * char_count & token_count
- Configurable target chunk size and overlap (default ~600 tokens, ~15% overlap).
- Save chunks to data/processed/<stem>_chunks.json.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("processing.chunker")

# Regex for identifying sections: e.g. "SECTION 7 — TERMINATION", "ARTICLE IV: RENT", "PART II"
RE_SECTION_HEADER = re.compile(
    r"^(?:SECTION|ARTICLE|PART|CHAPTER)\s+([0-9IVXLCDM]+)[\s:.\-—–]*(.*?)$",
    re.IGNORECASE,
)
# Numbered contract headings, including headings followed by body text.
# Examples: "4. EARLY TERMINATION." and "18. DISPUTES. This Agreement..."
RE_NUMBERED_SECTION_HEADER = re.compile(
    r"^(\d{1,2})\.\s+"
    r"([A-Z][A-Z0-9\s,&/()'’\-—–]{1,79}?)"
    r"(?:\.(?=\s|$)|(?=\s+[A-Z][a-z])|$)"
    r"(?:\s+.*)?$"
)
# Unnumbered uppercase headings, optionally followed by body text.
# Examples: "TERM. This Agreement..." and "EARLY TERMINATION."
RE_BARE_CAPS_HEADING = re.compile(
    r"^([A-Z][A-Z0-9\s,&/()'’\-—–]{1,79}?)\.(?:\s+.*)?$"
)
# Standalone uppercase section headers (e.g. "TERMINATION CONDITIONS", "OBLIGATIONS OF TENANT")
RE_CAPS_SECTION = re.compile(
    r"^([A-Z0-9\s,\-—–]{4,60})$"
)

# Regex for identifying clauses: e.g. "7.2", "Clause 4", "12.1.3", "(a)"
RE_CLAUSE_HEADER = re.compile(
    r"^(?:(?:CLAUSE|PARAGRAPH|RULE)\s+)?(\d+(?:\.\d+)+|\d+\.|\([a-z0-9ivx]+\))\s*(.*?)$",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """
    Fast, reliable token estimation for English/legal text.
    Standard approximation: ~1 token per 0.75 words, or ~4 chars per token.
    """
    if not text:
        return 0
    words = len(re.findall(r"\S+", text))
    return max(1, int(words * 1.3))


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while respecting common legal abbreviations
    (e.g., 'No.', 'Sec.', 'Art.', 'vs.', 'e.g.', 'i.e.', 'Ltd.', 'Inc.').
    """
    # Replace dots in known legal abbreviations temporarily with placeholder
    protected = text
    abbrevs = [
        r"No\.", r"Sec\.", r"Art\.", r"Cl\.", r"vs\.", r"v\.", r"e\.g\.", r"i\.e\.",
        r"Ltd\.", r"Inc\.", r"Co\.", r"Corp\.", r"para\.", r"sub\-s\.", r"approx\.",
    ]
    replacements = {}
    for i, pat in enumerate(abbrevs):
        placeholder = f"__ABBR_{i}__"
        matches = re.findall(pat, protected, re.IGNORECASE)
        for m in matches:
            replacements[placeholder] = m
            protected = protected.replace(m, placeholder)

    # Split on sentence boundaries: period, question mark, exclamation mark followed by space + uppercase
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\"'])", protected)

    sentences = []
    for s in raw_sents:
        # Restore placeholders
        for placeholder, original in replacements.items():
            s = s.replace(placeholder, original)
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences if sentences else [text.strip()]


@dataclass
class SemanticUnit:
    """A semantic unit of legal text (clause, paragraph, or section piece)."""

    text: str
    page_number: int
    section: Optional[str]
    clause: Optional[str]
    token_count: int


def _parse_page_semantic_units(
    page: Dict[str, Any],
    inherited_section: Optional[str] = None,
) -> Tuple[List[SemanticUnit], Optional[str]]:
    """
    Parse a cleaned page into semantic units while tracking sections and clauses.
    """
    p_num = page.get("page_number", 1)
    page_text = page.get("text", "")
    units: List[SemanticUnit] = []

    if not page_text or not page_text.strip():
        return units, inherited_section

    paragraphs = page_text.split("\n\n")
    current_section = inherited_section
    current_clause: Optional[str] = None
    pending_section_num: Optional[str] = None
    in_signature_block = False

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        first_line = para.splitlines()[0].strip()

        if re.search(r"\bIN WITNESS WHEREOF\b", para, re.IGNORECASE):
            in_signature_block = True
        if in_signature_block:
            units.append(
                SemanticUnit(
                    text=para,
                    page_number=p_num,
                    section=current_section,
                    clause=None,
                    token_count=estimate_tokens(para),
                )
            )
            continue

        # A standalone number may precede a section heading on the next
        # paragraph, as in "3." followed by "TERM. This Agreement...".
        standalone_num = re.fullmatch(r"(\d{1,2})\.", first_line)
        if standalone_num:
            pending_section_num = standalone_num.group(1)
            units.append(
                SemanticUnit(
                    text=para,
                    page_number=p_num,
                    section=current_section,
                    clause=current_clause,
                    token_count=estimate_tokens(para),
                )
            )
            continue

        # Detect explicit section headings.
        sec_match = RE_SECTION_HEADER.match(first_line)
        numbered_sec_match = RE_NUMBERED_SECTION_HEADER.match(first_line)
        bare_caps_match = RE_BARE_CAPS_HEADING.match(first_line)
        is_section_header = False

        if sec_match:
            sec_num = sec_match.group(1).strip()
            sec_title = sec_match.group(2).strip()
            current_section = (
                f"Section {sec_num}: {sec_title}" if sec_title
                else f"Section {sec_num}"
            )
            current_clause = None
            is_section_header = True

        elif numbered_sec_match:
            sec_num = numbered_sec_match.group(1)
            sec_title = numbered_sec_match.group(2).strip().rstrip(".")
            current_section = f"Section {sec_num}: {sec_title.title()}"
            current_clause = None
            is_section_header = True

        elif bare_caps_match:
            sec_title = bare_caps_match.group(1).strip().rstrip(".")
            if pending_section_num:
                current_section = f"Section {pending_section_num}: {sec_title.title()}"
                pending_section_num = None
            else:
                current_section = sec_title.title()
            current_clause = None
            is_section_header = True

        elif (
            RE_CAPS_SECTION.match(first_line)
            and len(first_line.split()) <= 8
            and not first_line.endswith(".")
        ):
            current_section = first_line.title()
            current_clause = None
            is_section_header = True

        # Only parse a clause number when this is not a section heading.
        if not is_section_header:
            cl_match = RE_CLAUSE_HEADER.match(first_line)
            if cl_match:
                current_clause = cl_match.group(1).strip().rstrip(".")

        unit = SemanticUnit(
            text=para,
            page_number=p_num,
            section=current_section,
            clause=current_clause,
            token_count=estimate_tokens(para),
        )
        units.append(unit)

    return units, current_section


@dataclass
class LegalChunk:
    """Metadata and content for a single legal chunk."""

    chunk_id: str
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    section: Optional[str]
    clause: Optional[str]
    text: str
    char_count: int
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_document(
    cleaned_doc: Dict[str, Any],
    target_tokens: int = 600,
    overlap_tokens: int = 90,
    max_tokens: int = 900,
) -> List[Dict[str, Any]]:
    """
    Split a cleaned document into legal-aware semantic chunks.

    Parameters
    ----------
    cleaned_doc : dict
        Output from cleaner.clean_document.
    target_tokens : int, optional
        Target chunk size in tokens (default 600).
    overlap_tokens : int, optional
        Target overlap when splitting across chunks (default 90).
    max_tokens : int, optional
        Hard maximum threshold beyond which a large unit must be broken down (default 900).

    Returns
    -------
    list of dict
        List of serialized LegalChunk dictionaries.
    """
    document_id = cleaned_doc.get("document_id", "doc")
    source_file = cleaned_doc.get("source_file", "document.pdf")
    pages = cleaned_doc.get("pages", [])

    if not pages:
        return []

    # 1. Parse all pages into semantic units with section/clause tracking
    all_units: List[SemanticUnit] = []
    current_section: Optional[str] = None

    for page in pages:
        page_units, current_section = _parse_page_semantic_units(
            page, inherited_section=current_section
        )
        all_units.extend(page_units)

    if not all_units:
        return []

    # 2. Group semantic units into chunks respecting boundaries
    chunks: List[LegalChunk] = []
    chunk_sequence = 1

    current_units: List[SemanticUnit] = []
    current_tokens = 0

    def flush_chunk(units_to_pack: List[SemanticUnit]) -> Optional[LegalChunk]:
        nonlocal chunk_sequence
        if not units_to_pack:
            return None

        combined_text = "\n\n".join(u.text for u in units_to_pack).strip()
        if not combined_text:
            return None

        p_start = min(u.page_number for u in units_to_pack)
        p_end = max(u.page_number for u in units_to_pack)

        # Retain section & clause from the first or dominant unit
        # Never invent section or clause
        sections = [u.section for u in units_to_pack if u.section]
        clauses = [u.clause for u in units_to_pack if u.clause]

        chunk_section = sections[0] if sections else None
        chunk_clause = clauses[0] if clauses else None

        cid = f"{document_id}_p{p_start:03d}_c{chunk_sequence:03d}"
        chunk_sequence += 1

        t_count = estimate_tokens(combined_text)

        return LegalChunk(
            chunk_id=cid,
            document_id=document_id,
            source_file=source_file,
            page_start=p_start,
            page_end=p_end,
            section=chunk_section,
            clause=chunk_clause,
            text=combined_text,
            char_count=len(combined_text),
            token_count=t_count,
        )

    for unit in all_units:
        # Case A: A single semantic unit is extraordinarily large (> max_tokens).
        # We must split it at sentence boundaries.
        if unit.token_count > max_tokens:
            # Flush whatever we have accumulated first
            if current_units:
                ch = flush_chunk(current_units)
                if ch:
                    chunks.append(ch)
                current_units = []
                current_tokens = 0

            # Split large unit by sentences
            sentences = split_into_sentences(unit.text)
            sub_sent_units: List[SemanticUnit] = []
            sub_tokens = 0

            for sent in sentences:
                s_tok = estimate_tokens(sent)
                if sub_tokens + s_tok > target_tokens and sub_sent_units:
                    ch = flush_chunk(sub_sent_units)
                    if ch:
                        chunks.append(ch)
                    # Handle overlap: retain last sentence if meaningful
                    if overlap_tokens > 0 and sub_sent_units:
                        last_u = sub_sent_units[-1]
                        sub_sent_units = [last_u]
                        sub_tokens = last_u.token_count
                    else:
                        sub_sent_units = []
                        sub_tokens = 0

                sub_sent_units.append(
                    SemanticUnit(
                        text=sent,
                        page_number=unit.page_number,
                        section=unit.section,
                        clause=unit.clause,
                        token_count=s_tok,
                    )
                )
                sub_tokens += s_tok

            if sub_sent_units:
                ch = flush_chunk(sub_sent_units)
                if ch:
                    chunks.append(ch)
            continue

        # Case B: Semantic boundary or size threshold reached.
        # Check if a new section starts, or if adding this unit would exceed target size.
        section_changed = (
            bool(current_units)
            and unit.section is not None
            and current_units[-1].section != unit.section
        )
        size_exceeded = (current_tokens + unit.token_count > target_tokens)

        if (section_changed or size_exceeded) and current_units:
            ch = flush_chunk(current_units)
            if ch:
                chunks.append(ch)

            # Apply semantic overlap if configured (only if size exceeded, not if clean section boundary)
            if size_exceeded and overlap_tokens > 0 and current_units:
                # Include the last unit as context overlap if it fits comfortably
                last_unit = current_units[-1]
                if last_unit.token_count <= overlap_tokens * 1.5:
                    current_units = [last_unit]
                    current_tokens = last_unit.token_count
                else:
                    current_units = []
                    current_tokens = 0
            else:
                current_units = []
                current_tokens = 0

        current_units.append(unit)
        current_tokens += unit.token_count

    # Flush final pending units
    if current_units:
        ch = flush_chunk(current_units)
        if ch:
            chunks.append(ch)

    logger.info(
        "Document '%s' chunked into %d chunks (target=%d tokens, overlap=%d tokens)",
        source_file,
        len(chunks),
        target_tokens,
        overlap_tokens,
    )

    return [c.to_dict() for c in chunks]


def save_chunks(
    chunks: List[Dict[str, Any]],
    source_file: str,
    output_dir: str | Path = "data/processed",
) -> Path:
    """Save chunks list to structured JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_file).stem
    out_path = out_dir / f"{stem}_chunks.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d chunks to: %s", len(chunks), out_path)
    return out_path
