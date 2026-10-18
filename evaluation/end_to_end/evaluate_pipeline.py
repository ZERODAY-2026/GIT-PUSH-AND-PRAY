"""
evaluate_pipeline.py
====================
Stage 3: Full End-to-End Pipeline Evaluation on CUAD Benchmark.

Tests your entire system exactly as a real user uses it:
  Raw Contract Text → Cleaning → Chunking → Embeddings → FAISS →
  LegalRetriever → Candidate Chunks → Gemma LLM → Final Triage Answer.

CRITICAL RULE:
  Gemma receives ONLY the evidence retrieved by your actual system.
  CUAD gold context is NEVER leaked to Gemma.

Classifies failure attribution:
  - EXTRACTION FAILURE
  - RETRIEVAL FAILURE
  - LLM FAILURE
  - GROUNDING FAILURE
  - TECHNICAL FAILURE
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Existing production components — untouched
from src.processing.cleaner import clean_document
from src.processing.chunker import chunk_document
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import LegalRetriever
from src.answer_question import answer_question
from src.generation.gemma_client import LocalGemmaClient

# Stage 3 evaluation modules
from evaluation.retrieval.matching import is_chunk_relevant
from evaluation.llm.metrics import compute_token_f1, evaluate_groundedness
from evaluation.llm.evaluate_llm import MockGemmaClient
from evaluation.end_to_end.metrics import classify_pipeline_result, compute_end_to_end_metrics
from evaluation.cuad.prepare_cuad import prepare_cuad_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.pipeline")

OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "end_to_end"


def run_pipeline_evaluation(
    num_contracts: int = 2,
    top_k: int = 5,
    min_score: float = 0.40,
    limit: int | None = 10,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 3 End-to-End Evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cuad_prep_file = BASE_DIR / "evaluation" / "cuad" / "data" / "cuad_processed.json"
    if not cuad_prep_file.exists():
        logger.info("Prepared CUAD data not found. Preparing now...")
        dataset = prepare_cuad_dataset()
    else:
        with open(cuad_prep_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    contracts = dataset.get("contracts", [])[:num_contracts]
    if not contracts:
        raise ValueError("No contracts found in dataset.")

    # Initialize Gemma LLM
    llm: Any = None
    if use_mock:
        logger.info("Using offline mock Gemma client (--mock specified).")
        llm = MockGemmaClient()
    else:
        client = LocalGemmaClient()
        conn = client.check_connection()
        if not conn.get("reachable"):
            logger.warning(
                f"\n⚠️  Local Gemma inference server at {client.base_url} is NOT reachable.\n"
                "Please make sure LM Studio or Ollama is running with Gemma loaded.\n"
                "To run the end-to-end evaluation pipeline offline, run with: --mock\n"
            )
            raise ConnectionError(
                f"Could not connect to local Gemma server at {client.base_url}. "
                "Start LM Studio/Ollama or pass `--mock` for offline testing."
            )
        llm = client

    embedder = LocalEmbedder()
    all_results: List[Dict[str, Any]] = []
    processed_count = 0

    for c_idx, contract_data in enumerate(contracts, 1):
        contract_title = contract_data.get("contract_title", f"Contract_{c_idx}")
        context_text = contract_data.get("context", "")
        examples = contract_data.get("examples", [])

        logger.info(f"[{c_idx}/{len(contracts)}] Indexing full contract: {contract_title[:50]}...")

        # 1. Document Extraction & Preprocessing Simulation
        page_size = 3000
        pages = []
        for p_no, start in enumerate(range(0, max(len(context_text), 1), page_size), 1):
            pages.append({"page_number": p_no, "text": context_text[start:start + page_size]})

        raw_doc = {"source_file": contract_title, "total_pages": len(pages), "pages": pages}

        # 2. Existing Cleaner & Chunker
        cleaned = clean_document(raw_doc)
        chunks = chunk_document(cleaned)

        # 3. Existing Embedder & FAISS Store
        embeddings = embedder.embed_chunks(chunks)
        dim = getattr(embedder, "dimension", 384)
        store = FAISSVectorStore(dimension=dim)
        store.add_documents(chunks, embeddings)
        retriever = LegalRetriever(embedder, store, top_k=top_k, min_score=min_score)

        for ex in examples:
            if limit is not None and processed_count >= limit:
                break

            category = ex.get("category", "")
            raw_question = ex.get("question", "")
            is_positive = ex.get("is_positive", True)
            gold_spans = [s["text"] for s in ex.get("gold_spans", [])]
            primary_gold = ex.get("gold_answer", "")

            logger.info(f"  [{processed_count + 1}] Evaluating End-to-End: '{category}'...")

            # 4. Run through existing answer_question orchestration
            start_t = time.time()
            technical_error = False
            try:
                pipeline_out = answer_question(category, retriever, llm)
                latency = round(time.time() - start_t, 2)
            except Exception as e:
                logger.error(f"Pipeline failure on {ex.get('example_id')}: {e}")
                pipeline_out = {
                    "question": category,
                    "status": "error",
                    "answer": f"ERROR: {e}",
                    "sources": [],
                    "best_score": None,
                }
                latency = 0.0
                technical_error = True

            retrieval_status = pipeline_out.get("status", "unknown")
            answer_text = pipeline_out.get("answer", "")
            sources = pipeline_out.get("sources", [])

            # Check if gold was among retrieved sources
            gold_found = False
            gold_rank = None
            for s_idx, src in enumerate(sources, 1):
                chunk_text = ""
                # Lookup chunk text from store
                cid = src.get("chunk_id")
                if cid and cid in store.chunks:
                    chunk_text = store.chunks[cid].get("text", "")
                is_rel, _, _ = is_chunk_relevant(chunk_text, gold_spans)
                if is_rel and not gold_found:
                    gold_found = True
                    gold_rank = s_idx

            silence_detected = None
            if not is_positive:
                silence_detected = (retrieval_status != "ok") or (len(sources) == 0)

            # Evaluate Answer against Gold
            f1_scores = compute_token_f1(answer_text, primary_gold) if primary_gold else {"f1": 1.0, "precision": 1.0, "recall": 1.0}
            grounded_scores = evaluate_groundedness(answer_text, primary_gold if primary_gold else "silent")

            classification = classify_pipeline_result(
                is_positive=is_positive,
                retrieval_status=retrieval_status,
                gold_found=gold_found,
                silence_detected=silence_detected,
                f1_score=f1_scores["f1"],
                groundedness_score=grounded_scores["groundedness_score"],
                unsupported_claim_rate=grounded_scores["unsupported_claim_rate"],
                technical_error=technical_error,
            )

            record = {
                "example_id": ex.get("example_id"),
                "document_id": contract_title,
                "category": category,
                "question": raw_question,
                "is_positive": is_positive,
                "gold_answer": primary_gold,
                "retrieval": {
                    "status": retrieval_status,
                    "gold_found": gold_found,
                    "gold_rank": gold_rank,
                    "silence_detected": silence_detected,
                    "sources_retrieved": len(sources),
                },
                "model_answer": answer_text,
                "evaluation": {
                    "status": classification["status"],
                    "failure_stage": classification["failure_stage"],
                    "explanation": classification["explanation"],
                    "f1_score": f1_scores["f1"],
                    "groundedness_score": grounded_scores["groundedness_score"],
                    "unsupported_claim_rate": grounded_scores["unsupported_claim_rate"],
                    "latency_sec": latency,
                },
            }

            all_results.append(record)
            processed_count += 1

        if limit is not None and processed_count >= limit:
            break

    # Aggregate Metrics
    e2e_metrics = compute_end_to_end_metrics(all_results)

    # 1. Save JSONL
    jsonl_path = OUTPUT_DIR / "end_to_end_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Save CSV
    csv_path = OUTPUT_DIR / "end_to_end_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_id", "document_id", "category", "is_positive",
            "gold_found", "gold_rank", "pipeline_status", "failure_stage", "f1_score", "groundedness_score"
        ])
        for r in all_results:
            writer.writerow([
                r["example_id"],
                r["document_id"][:40],
                r["category"],
                r["is_positive"],
                r["retrieval"]["gold_found"],
                r["retrieval"]["gold_rank"] or "",
                r["evaluation"]["status"],
                r["evaluation"]["failure_stage"],
                r["evaluation"]["f1_score"],
                r["evaluation"]["groundedness_score"],
            ])

    # 3. Save JSON Report
    json_rep_path = OUTPUT_DIR / "end_to_end_report.json"
    with open(json_rep_path, "w", encoding="utf-8") as f:
        json.dump(e2e_metrics, f, indent=2, ensure_ascii=False)

    # 4. Save Markdown Report
    md_rep_path = OUTPUT_DIR / "end_to_end_report.md"
    generate_e2e_markdown_report(all_results, e2e_metrics, md_rep_path)

    logger.info(
        f"\n=======================================================\n"
        f"Stage 3: End-to-End Pipeline Evaluation Completed!\n"
        f"  Total Examples:        {e2e_metrics.get('total_examples_evaluated')}\n"
        f"  Overall Accuracy:      {e2e_metrics.get('overall_end_to_end_accuracy') * 100:.1f}%\n"
        f"  Retrieval Recall:      {e2e_metrics.get('positive_clause_recall') * 100:.1f}%\n"
        f"  Silence Accuracy:      {e2e_metrics.get('silence_accuracy') * 100:.1f}%\n"
        f"  Failure Breakdown:     {e2e_metrics.get('failure_stage_distribution')}\n"
        f"Saved results to {OUTPUT_DIR}\n"
        f"======================================================="
    )
    return e2e_metrics


def generate_e2e_markdown_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate Markdown report for Stage 3 end-to-end evaluation."""
    lines = [
        "# Stage 3 — CUAD End-to-End Pipeline Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This evaluation tests your **complete production pipeline** exactly as a user experiences it: from raw contract text processing and FAISS indexing, through thresholded retrieval, to local Gemma triage response generation.",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
        f"| **Total Evaluated** | `{metrics.get('total_examples_evaluated', 0)}` | End-to-end queries processed |",
        f"| **Overall End-to-End Accuracy** | **`{metrics.get('overall_end_to_end_accuracy', 0.0) * 100:.1f}%`** | Successful retrieval AND grounded answer |",
        f"| **Positive Clause Recall** | **`{metrics.get('positive_clause_recall', 0.0) * 100:.1f}%`** | Found gold clause in retrieved chunks |",
        f"| **Silence Detection Rate** | **`{metrics.get('silence_accuracy', 0.0) * 100:.1f}%`** | Successfully suppressed false positive retrieval |",
        "",
        "---",
        "",
        "## Error Analysis & Failure Attribution",
        "",
        "| Failure Category | Occurrences | Description |",
        "|---|---:|---|",
    ]

    failures = metrics.get("failure_stage_distribution", {})
    descriptions = {
        "EXTRACTION FAILURE": "Raw text was not properly segmented into searchable blocks",
        "RETRIEVAL FAILURE": "Target clause was missed during dense vector cosine retrieval",
        "LLM FAILURE": "Correct chunk was retrieved but Gemma failed to extract the answer",
        "GROUNDING FAILURE": "Gemma introduced unverified facts not supported by the chunk",
        "TECHNICAL FAILURE": "Exception during execution (memory, timeout, index error)",
        "UNKNOWN / REQUIRES REVIEW": "Edge cases requiring human legal expert inspection",
    }

    for cat, count in failures.items():
        desc = descriptions.get(cat, "")
        lines.append(f"| **`{cat}`** | **`{count}`** | {desc} |")

    lines.extend([
        "",
        "---",
        "",
        "## Category-wise End-to-End Performance",
        "",
        "| Category | Total Tested | Success Rate | Retrieval Recall |",
        "|---|---:|---:|---:|",
    ])

    breakdown = metrics.get("category_breakdown", {})
    for cat, b in sorted(breakdown.items(), key=lambda x: x[0]):
        lines.append(
            f"| **{cat}** | {b['total_examples']} | "
            f"**{b['success_rate'] * 100:.1f}%** | {b['retrieval_recall'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Sample Successful Case Study",
        "",
    ])

    successes = [r for r in results if r.get("evaluation", {}).get("status") == "SUCCESS"]
    if successes:
        s = successes[0]
        lines.extend([
            f"### Category: `{s['category']}`",
            f"- **Contract**: {s['document_id']}",
            f"- **Gold Span**: > \"{s['gold_answer'][:250]}...\"",
            f"- **Gemma Output (Snippet)**:\n\n```markdown\n{s['model_answer'][:400]}...\n```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Sample Failure Case Study & Diagnosis",
        "",
    ])

    failed_cases = [r for r in results if r.get("evaluation", {}).get("status") == "FAILURE"]
    if failed_cases:
        f = failed_cases[0]
        lines.extend([
            f"### Failed Category: `{f['category']}`",
            f"- **Contract**: {f['document_id']}",
            f"- **Failure Stage**: `{f['evaluation']['failure_stage']}`",
            f"- **Diagnosis**: {f['evaluation']['explanation']}",
            f"- **Gold Required**: > \"{f['gold_answer'][:250]}...\"",
            "",
        ])
    else:
        lines.append("*No failures encountered in evaluated sample.*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: End-to-end evaluation of full pipeline on CUAD.")
    parser.add_argument("--num_contracts", type=int, default=2, help="Number of contracts to evaluate")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K chunks for retriever")
    parser.add_argument("--min_score", type=float, default=0.40, help="Retrieval cosine score cutoff")
    parser.add_argument("--limit", type=int, default=10, help="Total queries limit for fast smoke testing")
    parser.add_argument("--mock", action="store_true", help="Use offline mock Gemma client if inference server is offline")
    args = parser.parse_args()

    run_pipeline_evaluation(
        num_contracts=args.num_contracts,
        top_k=args.top_k,
        min_score=args.min_score,
        limit=args.limit,
        use_mock=args.mock,
    )
