"""
prepare_cuad.py
===============
Loads and normalizes the raw CUAD dataset (SQuAD 2.0 format) into structured,
reproducible benchmark evaluation records for the legal triage AI system.

Preserves:
- Document / Contract ID and title
- Clause category (normalized from standard Atticus 41 categories)
- Full question prompt and category short name
- Raw contract text (for indexing and chunking)
- Ground-truth clause answers with character offsets
- Silence flag (`is_impossible = True` when contract contains no such provision)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cuad.prepare")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CUAD_RAW_DIR = BASE_DIR / "backend" / "data" / "cuad"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def extract_category_from_question(question: str) -> str:
    """Extract standard category name enclosed in quotes or clean phrase."""
    # Pattern: Highlight the parts ... related to "Category Name" that should ...
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1).strip()
    return question.strip()


def load_category_descriptions() -> Dict[str, str]:
    """Load the official CUAD 41 category descriptions if available."""
    desc_csv = CUAD_RAW_DIR / "category_descriptions.csv"
    descriptions: Dict[str, str] = {}
    if desc_csv.exists():
        try:
            with open(desc_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        cat_name = row[0].strip()
                        cat_desc = row[1].strip()
                        descriptions[cat_name] = cat_desc
        except Exception as e:
            logger.warning(f"Could not load category descriptions: {e}")
    return descriptions


def prepare_cuad_dataset(
    source_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    max_contracts: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse raw CUAD json into standardized benchmark records."""
    if source_file is None:
        test_path = CUAD_RAW_DIR / "test.json"
        cuad_v1 = CUAD_RAW_DIR / "CUADv1.json"
        source_file = test_path if test_path.exists() else cuad_v1

    if not source_file.exists():
        raise FileNotFoundError(
            f"CUAD dataset file not found at {source_file}. "
            "Please ensure CUADv1.json or test.json is in backend/data/cuad/"
        )

    logger.info(f"Loading raw CUAD from: {source_file.name}...")
    with open(source_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    contracts = raw_data.get("data", [])
    if max_contracts:
        contracts = contracts[:max_contracts]

    logger.info(f"Processing {len(contracts)} contracts...")
    cat_descriptions = load_category_descriptions()

    prepared_contracts: List[Dict[str, Any]] = []
    total_examples = 0
    total_positive = 0
    total_silence = 0
    category_counts: Dict[str, Dict[str, int]] = {}

    for c_idx, contract in enumerate(contracts, 1):
        contract_title = contract.get("title", f"contract_{c_idx}")
        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue

        p0 = paragraphs[0]
        context_text = p0.get("context", "")
        qas = p0.get("qas", [])

        contract_examples: List[Dict[str, Any]] = []

        for q_idx, qa in enumerate(qas, 1):
            qa_id = qa.get("id", f"cuad_{c_idx}_{q_idx}")
            raw_question = qa.get("question", "")
            category = extract_category_from_question(raw_question)
            is_impossible = bool(qa.get("is_impossible", False))
            answers = qa.get("answers", [])

            gold_spans = []
            for ans in answers:
                text = ans.get("text", "").strip()
                start = ans.get("answer_start", -1)
                if text:
                    gold_spans.append({"text": text, "answer_start": start})

            is_positive = (not is_impossible) and (len(gold_spans) > 0)
            if is_positive:
                total_positive += 1
            else:
                total_silence += 1
            total_examples += 1

            if category not in category_counts:
                category_counts[category] = {"positive": 0, "silence": 0, "total": 0}
            category_counts[category]["total"] += 1
            if is_positive:
                category_counts[category]["positive"] += 1
            else:
                category_counts[category]["silence"] += 1

            example = {
                "example_id": qa_id,
                "contract_index": c_idx,
                "contract_title": contract_title,
                "category": category,
                "category_description": cat_descriptions.get(category, ""),
                "question": raw_question,
                "is_impossible": is_impossible,
                "is_positive": is_positive,
                "gold_spans": gold_spans,
                "gold_answer": gold_spans[0]["text"] if gold_spans else "",
                "all_gold_answers": [s["text"] for s in gold_spans],
            }
            contract_examples.append(example)

        prepared_contracts.append({
            "contract_index": c_idx,
            "contract_title": contract_title,
            "context_length": len(context_text),
            "context": context_text,
            "examples": contract_examples,
        })

    dataset_summary = {
        "version": raw_data.get("version", "1.0"),
        "source_file": source_file.name,
        "total_contracts": len(prepared_contracts),
        "total_examples": total_examples,
        "total_positive_examples": total_positive,
        "total_silence_examples": total_silence,
        "total_categories": len(category_counts),
        "categories": category_counts,
        "contracts": prepared_contracts,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_file or (OUTPUT_DIR / "cuad_processed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"CUAD dataset prepared successfully! Saved to: {out_path}\n"
        f"  - Contracts: {len(prepared_contracts)}\n"
        f"  - Total Examples: {total_examples} ({total_positive} positive, {total_silence} silence)\n"
        f"  - Categories: {len(category_counts)}"
    )
    return dataset_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and normalize CUAD benchmark dataset.")
    parser.add_argument("--max_contracts", type=int, default=None, help="Limit number of contracts to process")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    prepare_cuad_dataset(output_file=out_p, max_contracts=args.max_contracts)
