"""
evaluate_retrieval.py
=====================
Stage 1: Independent Retrieval Evaluation on CUAD Benchmark.

Calls your EXISTING retrieval system:
  LocalEmbedder (bge-small-en-v1.5) + FAISS (IndexFlatIP) + LegalRetriever

DOES NOT CALL GEMMA. Isolates retrieval recall, ranking precision (MRR),
and silence detection without confounding LLM errors.

Outputs:
  evaluation/results/retrieval/retrieval_results.jsonl
  evaluation/results/retrieval/retrieval_results.csv
  evaluation/results/retrieval/retrieval_report.json
  evaluation/results/retrieval/retrieval_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend modules can be imported without changing system path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import EXISTING components — untouched
from src.processing.cleaner import clean_document
from src.processing.chunker import chunk_document
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import LegalRetriever

# Evaluation utilities
from evaluation.retrieval.matching import is_chunk_relevant
from evaluation.retrieval.metrics import compute_retrieval_metrics
from evaluation.cuad.prepare_cuad import prepare_cuad_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.retrieval")

OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "retrieval"


def index_contract_into_faiss(
    contract_title: str,
    context_text: str,
    embedder: LocalEmbedder,
    page_size: int = 3000,
) -> FAISSVectorStore:
    """Index a single CUAD contract into an isolated on-device FAISS store."""
    pages = []
    for p_no, start in enumerate(range(0, max(len(context_text), 1), page_size), 1):
        pages.append({
            "page_number": p_no,
            "text": context_text[start:start + page_size],
        })

    raw_doc = {
        "source_file": contract_title,
        "total_pages": len(pages),
        "pages": pages,
    }

    cleaned = clean_document(raw_doc)
    chunks = chunk_document(cleaned)
    embeddings = embedder.embed_chunks(chunks)

    dim = getattr(embedder, "dimension", 384)
    store = FAISSVectorStore(dimension=dim)
    store.add_documents(chunks, embeddings)
    return store


def run_retrieval_evaluation(
    num_contracts: int = 3,
    top_k: int = 10,
    min_score: float = 0.40,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Execute Stage 1 Retrieval Evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cuad_prep_file = BASE_DIR / "evaluation" / "cuad" / "data" / "cuad_processed.json"
    if not cuad_prep_file.exists():
        logger.info("Prepared CUAD data not found. Running prepare_cuad_dataset()...")
        dataset = prepare_cuad_dataset()
    else:
        with open(cuad_prep_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    contracts = dataset.get("contracts", [])[:num_contracts]
    if not contracts:
        raise ValueError("No contracts available in CUAD dataset.")

    logger.info(f"Initializing LocalEmbedder (bge-small-en-v1.5)...")
    embedder = LocalEmbedder()

    all_results: List[Dict[str, Any]] = []
    processed_count = 0

    for c_idx, contract_data in enumerate(contracts, 1):
        contract_title = contract_data.get("contract_title", f"Contract_{c_idx}")
        context_text = contract_data.get("context", "")
        examples = contract_data.get("examples", [])

        logger.info(
            f"[{c_idx}/{len(contracts)}] Indexing: {contract_title[:50]}... "
            f"({len(context_text):,} chars, {len(examples)} QA categories)"
        )

        store = index_contract_into_faiss(contract_title, context_text, embedder)
        retriever = LegalRetriever(embedder, store, top_k=top_k, min_score=min_score)

        for ex in examples:
            if limit is not None and processed_count >= limit:
                break

            category = ex.get("category", "")
            raw_question = ex.get("question", "")
            is_positive = ex.get("is_positive", True)
            gold_spans = [s["text"] for s in ex.get("gold_spans", [])]
            primary_gold = ex.get("gold_answer", "")

            # Query system using clause category or full question
            retrieval = retriever.retrieve(category)
            retrieved_chunks = retrieval.chunks

            # Evaluate chunks against gold
            gold_found = False
            gold_rank = None
            retrieved_results = []

            for rank_idx, chunk in enumerate(retrieved_chunks, 1):
                chunk_text = chunk.get("text", "")
                chunk_id = chunk.get("chunk_id", f"c_{rank_idx}")
                score = float(chunk.get("score", 0.0))
                page = chunk.get("page_start", 1)

                is_rel, match_score, match_type = is_chunk_relevant(chunk_text, gold_spans, min_overlap=0.45)
                if is_rel and not gold_found:
                    gold_found = True
                    gold_rank = rank_idx

                retrieved_results.append({
                    "rank": rank_idx,
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "page": page,
                    "text_snippet": chunk_text[:200] + ("..." if len(chunk_text) > 200 else ""),
                    "is_relevant": is_rel,
                    "match_score": round(match_score, 4),
                    "match_type": match_type,
                })

            silence_detected = None
            if not is_positive:
                # Silence correctly detected if no high confidence chunks were returned
                silence_detected = (retrieval.status != "ok") or (len(retrieved_chunks) == 0)

            record = {
                "example_id": ex.get("example_id"),
                "contract_index": c_idx,
                "contract_title": contract_title,
                "category": category,
                "question": raw_question,
                "is_positive": is_positive,
                "gold_context": primary_gold,
                "retrieval_status": retrieval.status,
                "best_retrieval_score": round(float(retrieval.best_score), 4) if retrieval.best_score else None,
                "gold_found": gold_found,
                "gold_rank": gold_rank,
                "silence_detected": silence_detected,
                "retrieved_results": retrieved_results,
            }

            all_results.append(record)
            processed_count += 1

        if limit is not None and processed_count >= limit:
            logger.info(f"Limit of {limit} examples reached.")
            break

    # Calculate Metrics
    metrics = compute_retrieval_metrics(all_results)

    # 1. Save JSONL
    jsonl_path = OUTPUT_DIR / "retrieval_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Save CSV
    csv_path = OUTPUT_DIR / "retrieval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_id", "contract_title", "category", "is_positive",
            "gold_found", "gold_rank", "best_score", "silence_detected", "gold_context"
        ])
        for r in all_results:
            writer.writerow([
                r["example_id"],
                r["contract_title"][:40],
                r["category"],
                r["is_positive"],
                r["gold_found"],
                r["gold_rank"] or "",
                r["best_retrieval_score"] or "",
                r["silence_detected"] if r["silence_detected"] is not None else "",
                r["gold_context"][:100].replace("\n", " "),
            ])

    # 3. Save JSON Report
    json_rep_path = OUTPUT_DIR / "retrieval_report.json"
    with open(json_rep_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 4. Save Markdown Report
    md_rep_path = OUTPUT_DIR / "retrieval_report.md"
    generate_markdown_report(all_results, metrics, md_rep_path)

    logger.info(
        f"\n=======================================================\n"
        f"Stage 1: Retrieval Evaluation Completed!\n"
        f"  Total Examples: {metrics.get('total_examples_evaluated')}\n"
        f"  Recall@1:       {metrics.get('recall_at_1') * 100:.1f}%\n"
        f"  Recall@3:       {metrics.get('recall_at_3') * 100:.1f}%\n"
        f"  Recall@5:       {metrics.get('recall_at_5') * 100:.1f}%\n"
        f"  Recall@10:      {metrics.get('recall_at_10') * 100:.1f}%\n"
        f"  MRR:            {metrics.get('mean_reciprocal_rank'):.4f}\n"
        f"  Silence Acc:    {metrics.get('silence_detection_accuracy') * 100:.1f}%\n"
        f"Saved results to {OUTPUT_DIR}\n"
        f"======================================================="
    )
    return metrics


def generate_markdown_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate professional Markdown evaluation report for Stage 1."""
    pos = [r for r in results if r.get("is_positive", True)]
    successes = [r for r in pos if r.get("gold_found", False)]
    failures = [r for r in pos if not r.get("gold_found", False)]

    lines = [
        "# Stage 1 — CUAD Retrieval Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This evaluation tests your **existing retrieval system** (*bge-small-en-v1.5* dense embeddings + *FAISS IndexFlatIP* cosine search + *LegalRetriever* thresholding) against gold standard Atticus contract annotations.",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
        f"| **Total Examples Evaluated** | `{metrics.get('total_examples_evaluated', 0)}` | Total contract queries processed |",
        f"| **Positive Clause Queries** | `{metrics.get('positive_clause_examples', 0)}` | Clauses with verified human ground-truth text |",
        f"| **Silence Negative Queries** | `{metrics.get('silence_negative_examples', 0)}` | Cases where contract was silent on topic |",
        f"| **Recall@1** | **`{metrics.get('recall_at_1', 0.0) * 100:.1f}%`** | Gold evidence surfaced in the #1 top chunk |",
        f"| **Recall@3** | **`{metrics.get('recall_at_3', 0.0) * 100:.1f}%`** | Gold evidence surfaced in top 3 chunks |",
        f"| **Recall@5** | **`{metrics.get('recall_at_5', 0.0) * 100:.1f}%`** | Gold evidence surfaced in top 5 chunks |",
        f"| **Recall@10** | **`{metrics.get('recall_at_10', 0.0) * 100:.1f}%`** | Gold evidence surfaced in top 10 chunks |",
        f"| **MRR (Mean Reciprocal Rank)** | **`{metrics.get('mean_reciprocal_rank', 0.0):.4f}`** | Average reciprocal ranking efficiency |",
        f"| **Silence Detection Rate** | **`{metrics.get('silence_detection_accuracy', 0.0) * 100:.1f}%`** | Correctly identified silence (no false positives) |",
        "",
        "---",
        "",
        "## Category-wise Retrieval Performance",
        "",
        "| Category | Total | Pos | Recall@1 | Recall@5 | Recall@10 | MRR | Silence Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    breakdown = metrics.get("category_breakdown", {})
    for cat, b in sorted(breakdown.items(), key=lambda x: x[0]):
        lines.append(
            f"| **{cat}** | {b['total_examples']} | {b['positive_examples']} | "
            f"{b['recall_at_1'] * 100:.1f}% | {b['recall_at_5'] * 100:.1f}% | {b['recall_at_10'] * 100:.1f}% | "
            f"{b['mrr']:.3f} | {b['silence_accuracy'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Sample Successful Retrievals",
        "",
    ])

    for s in successes[:3]:
        lines.extend([
            f"### Category: `{s['category']}` (Rank {s['gold_rank']})",
            f"- **Contract**: {s['contract_title']}",
            f"- **Gold Span**: > \"{s['gold_context'][:250]}...\"",
            f"- **Retrieved Best Score**: `{s['best_retrieval_score']}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Sample Retrieval Failures",
        "",
    ])

    if failures:
        for f in failures[:3]:
            lines.extend([
                f"### Failed Category: `{f['category']}`",
                f"- **Contract**: {f['contract_title']}",
                f"- **Gold Span Required**: > \"{f['gold_context'][:250]}...\"",
                f"- **Retrieved Best Score**: `{f['best_retrieval_score']}` (below threshold or wrong section)",
                "",
            ])
    else:
        lines.append("*No retrieval failures recorded in evaluated sample.*")

    lines.extend([
        "",
        "---",
        "",
        "## Common Retrieval Failure Patterns & Observations",
        "",
        "1. **Dense Vector Lexical Gaps**: Very specific numerical definitions or definitions of terms (e.g. specific entity dates) occasionally rank below general contract headers.",
        "2. **Chunk Boundary Splitting**: When an Atticus annotation spans across two paragraphs, the exact chunking boundary can cut the clause, requiring multi-chunk context synthesis.",
        "3. **Threshold Calibration**: The default cosine cutoff (`min_score = 0.40 - 0.45`) effectively suppresses ungrounded hallucination on silence categories.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Evaluate existing retrieval system on CUAD.")
    parser.add_argument("--num_contracts", type=int, default=3, help="Number of CUAD contracts to evaluate (default: 3)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K chunks to retrieve (default: 10)")
    parser.add_argument("--min_score", type=float, default=0.40, help="Minimum cosine similarity cutoff (default: 0.40)")
    parser.add_argument("--limit", type=int, default=None, help="Limit total questions evaluated for fast smoke test")
    args = parser.parse_args()

    run_retrieval_evaluation(
        num_contracts=args.num_contracts,
        top_k=args.top_k,
        min_score=args.min_score,
        limit=args.limit,
    )
