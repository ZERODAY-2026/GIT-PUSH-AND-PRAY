"""
evaluate_llm.py
===============
Stage 2: Independent Gemma LLM Comprehension Evaluation on CUAD Benchmark.

Calls your EXISTING generation system:
  prompt_builder.build_messages + LocalGemmaClient

ISOLATES THE LLM FROM RETRIEVAL ERRORS:
  Provides Gemma with the verified CUAD gold clause context directly.
  Tests whether Gemma accurately summarizes, extracts, and reasons over
  ground-truth legal evidence without introducing unsupported hallucinations.

Outputs:
  evaluation/results/llm/llm_results.jsonl
  evaluation/results/llm/llm_results.csv
  evaluation/results/llm/llm_report.json
  evaluation/results/llm/llm_report.md
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

# Existing production components
from src.generation.prompt_builder import build_messages
from src.generation.gemma_client import LocalGemmaClient

# Stage 2 metrics
from evaluation.llm.metrics import (
    compute_exact_match,
    compute_token_f1,
    evaluate_groundedness,
    compute_aggregate_llm_metrics,
)
from evaluation.cuad.prepare_cuad import prepare_cuad_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.llm")

OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "llm"


class MockGemmaClient:
    """Deterministic offline fallback if local LM Studio is not active during CI/testing."""
    def __init__(self, model_name: str = "mock-gemma-offline"):
        self.model = model_name

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt_content = messages[-1].get("content", "")
        # Extract quoted excerpt from prompt
        return (
            "### 1. Explicit Document Statements\n"
            f"- Based on the provided document excerpts, the clause states: {prompt_content[-200:]} [SOURCE 1].\n\n"
            "### 2. Potential Interpretations & Practical Concerns\n"
            "- The language defines specific rights and obligations applicable to the counterparty.\n\n"
            "### 3. Jurisdiction & Enforceability Notice\n"
            "- Enforceability remains subject to governing state law and local precedent.\n\n"
            "### 4. Recommended Next Steps\n"
            "- Verify counterparty signatures and consult licensed legal counsel.\n\n"
            "Not legal advice; preliminary document triage only."
        )


def run_llm_evaluation(
    limit: int | None = 10,
    use_mock: bool = False,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Execute Stage 2 LLM Evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cuad_prep_file = BASE_DIR / "evaluation" / "cuad" / "data" / "cuad_processed.json"
    if not cuad_prep_file.exists():
        logger.info("Prepared CUAD data not found. Preparing now...")
        dataset = prepare_cuad_dataset()
    else:
        with open(cuad_prep_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    # Gather positive examples with verified gold context
    positive_examples: List[Dict[str, Any]] = []
    for contract in dataset.get("contracts", []):
        for ex in contract.get("examples", []):
            if ex.get("is_positive") and ex.get("gold_answer"):
                positive_examples.append(ex)

    if not positive_examples:
        raise ValueError("No positive CUAD clause examples found for LLM evaluation.")

    eval_examples = positive_examples[:limit] if limit else positive_examples
    logger.info(f"Evaluating {len(eval_examples)} positive CUAD examples on Gemma...")

    # Initialize client
    llm: Any = None
    if use_mock:
        logger.info("Using offline mock Gemma client (--mock specified).")
        llm = MockGemmaClient()
    else:
        client = LocalGemmaClient(temperature=temperature)
        conn = client.check_connection()
        if not conn.get("reachable"):
            logger.warning(
                f"\n⚠️  Local Gemma inference server at {client.base_url} is NOT reachable.\n"
                "Please make sure LM Studio or Ollama is running with Gemma loaded.\n"
                "To test the evaluation pipeline without a running server, re-run with: --mock\n"
            )
            # Check if user specifically requested live-only
            raise ConnectionError(
                f"Could not connect to local Gemma server at {client.base_url}. "
                "Start LM Studio/Ollama or run with `python evaluation/llm/evaluate_llm.py --mock` for offline testing."
            )
        logger.info(f"Connected to local Gemma server. Available models: {conn.get('models')}")
        llm = client

    results: List[Dict[str, Any]] = []
    model_name = getattr(llm, "model", "local-gemma")

    for idx, ex in enumerate(eval_examples, 1):
        example_id = ex.get("example_id")
        category = ex.get("category", "")
        question = ex.get("question", "")
        gold_context = ex.get("gold_answer", "")
        contract_title = ex.get("contract_title", "")

        logger.info(f"[{idx}/{len(eval_examples)}] Testing Gemma on category: '{category}'...")

        # Construct synthetic gold chunk to pass to build_messages
        synthetic_chunk = {
            "chunk_id": f"gold_chunk_{idx}",
            "source_file": contract_title,
            "page_start": 1,
            "page_end": 1,
            "section": category,
            "clause": category,
            "score": 1.0,
            "text": gold_context,
        }

        # Existing prompt builder
        messages = build_messages(question, [synthetic_chunk])

        start_time = time.time()
        try:
            gemma_response = llm.generate(messages)
            latency_sec = round(time.time() - start_time, 2)
        except Exception as e:
            logger.error(f"Inference error on {example_id}: {e}")
            gemma_response = f"ERROR: {e}"
            latency_sec = 0.0

        # Metrics calculation
        em = compute_exact_match(gemma_response, gold_context)
        f1_metrics = compute_token_f1(gemma_response, gold_context)
        grounded_metrics = evaluate_groundedness(gemma_response, gold_context)

        record = {
            "example_id": example_id,
            "category": category,
            "contract_title": contract_title,
            "question": question,
            "gold_context": gold_context,
            "model_answer": gemma_response,
            "model_name": model_name,
            "latency_sec": latency_sec,
            "exact_match": em,
            "f1": f1_metrics["f1"],
            "precision": f1_metrics["precision"],
            "recall": f1_metrics["recall"],
            "groundedness_score": grounded_metrics["groundedness_score"],
            "unsupported_claim_rate": grounded_metrics["unsupported_claim_rate"],
        }
        results.append(record)

    # Compute aggregate metrics
    agg_metrics = compute_aggregate_llm_metrics(results)
    agg_metrics["model_name"] = model_name

    # 1. Save JSONL
    jsonl_path = OUTPUT_DIR / "llm_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Save CSV
    csv_path = OUTPUT_DIR / "llm_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_id", "category", "contract_title", "f1",
            "precision", "recall", "groundedness_score", "unsupported_claim_rate", "latency_sec"
        ])
        for r in results:
            writer.writerow([
                r["example_id"],
                r["category"],
                r["contract_title"][:40],
                r["f1"],
                r["precision"],
                r["recall"],
                r["groundedness_score"],
                r["unsupported_claim_rate"],
                r["latency_sec"],
            ])

    # 3. Save JSON Report
    json_rep_path = OUTPUT_DIR / "llm_report.json"
    with open(json_rep_path, "w", encoding="utf-8") as f:
        json.dump(agg_metrics, f, indent=2, ensure_ascii=False)

    # 4. Save Markdown Report
    md_rep_path = OUTPUT_DIR / "llm_report.md"
    generate_llm_markdown_report(results, agg_metrics, md_rep_path)

    logger.info(
        f"\n=======================================================\n"
        f"Stage 2: LLM Comprehension Evaluation Completed!\n"
        f"  Total Examples:        {agg_metrics.get('total_examples_evaluated')}\n"
        f"  Model Tested:          {model_name}\n"
        f"  Mean Token F1:         {agg_metrics.get('mean_token_f1') * 100:.1f}%\n"
        f"  Key-Term Recall:       {agg_metrics.get('mean_key_term_recall') * 100:.1f}%\n"
        f"  Mean Groundedness:     {agg_metrics.get('mean_groundedness_score') * 100:.1f}%\n"
        f"  Unsupported Claim Rate:{agg_metrics.get('mean_unsupported_claim_rate') * 100:.1f}%\n"
        f"Saved results to {OUTPUT_DIR}\n"
        f"======================================================="
    )
    return agg_metrics


def generate_llm_markdown_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate Markdown report for Stage 2 LLM evaluation."""
    lines = [
        "# Stage 2 — CUAD LLM Comprehension Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This evaluation tests **Google Gemma** (via `LocalGemmaClient` and `prompt_builder.build_messages`) in **isolation from retrieval** by supplying verified CUAD gold contract evidence directly.",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
        f"| **Model Tested** | `{metrics.get('model_name')}` | On-device inference model |",
        f"| **Total Examples Evaluated** | `{metrics.get('total_examples_evaluated')}` | Positive CUAD clause questions |",
        f"| **Mean Token F1** | **`{metrics.get('mean_token_f1', 0.0) * 100:.1f}%`** | Word overlap between Gemma answer and gold clause |",
        f"| **Key-Term Recall** | **`{metrics.get('mean_key_term_recall', 0.0) * 100:.1f}%`** | Preservation of critical operative terms |",
        f"| **Groundedness Score** | **`{metrics.get('mean_groundedness_score', 0.0) * 100:.1f}%`** | Claims directly substantiated by provided text |",
        f"| **Unsupported Claim Rate** | **`{metrics.get('mean_unsupported_claim_rate', 0.0) * 100:.1f}%`** | Extraneous facts introduced (hallucinations) |",
        "",
        "---",
        "",
        "## Category-wise LLM Performance",
        "",
        "| Category | Examples | Mean F1 | Groundedness | Unsupported Rate |",
        "|---|---:|---:|---:|---:|",
    ]

    breakdown = metrics.get("category_breakdown", {})
    for cat, b in sorted(breakdown.items(), key=lambda x: x[0]):
        lines.append(
            f"| **{cat}** | {b['total_examples']} | {b['avg_f1'] * 100:.1f}% | "
            f"{b['avg_groundedness'] * 100:.1f}% | {b['avg_unsupported_rate'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Sample Grounded Generations",
        "",
    ])

    for r in results[:2]:
        lines.extend([
            f"### Category: `{r['category']}`",
            f"- **Gold Evidence**: > \"{r['gold_context'][:200]}...\"",
            f"- **Gemma Output (Snippet)**:\n\n```markdown\n{r['model_answer'][:400]}...\n```",
            f"- **Scores**: F1 = `{r['f1']}`, Groundedness = `{r['groundedness_score']}`, Latency = `{r['latency_sec']}s`",
            "",
        ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Evaluate Gemma comprehension on CUAD gold evidence.")
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to evaluate (default: 10)")
    parser.add_argument("--mock", action="store_true", help="Use offline mock client if local inference server is inactive")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    args = parser.parse_args()

    run_llm_evaluation(
        limit=args.limit,
        use_mock=args.mock,
        temperature=args.temperature,
    )
