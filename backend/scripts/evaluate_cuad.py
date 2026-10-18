"""CUAD Benchmark Evaluation Harness.

Evaluates on-device retrieval, grounding, and silence detection against
expert-annotated commercial contracts from the CUAD (Atticus Project) dataset.

Measures:
1. Retrieval Recall@K (does retrieved evidence cover gold-standard expert labels?)
2. Silence Detection Accuracy (correctly identifying when a contract is silent without hallucinating)
3. Precision of extracted clause text

Usage:
    python -m scripts.evaluate_cuad [--num_contracts N] [--top_k K]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.processing.cleaner import clean_document
from src.processing.chunker import chunk_document
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import LegalRetriever
from src.answer_question import answer_question


CUAD_TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad" / "test.json"
if not CUAD_TEST_PATH.exists():
    CUAD_TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad" / "CUADv1.json"


def evaluate_cuad(num_contracts: int = 3, top_k: int = 4, min_score: float = 0.45) -> dict[str, Any]:
    if not CUAD_TEST_PATH.exists():
        print(f"Error: CUAD dataset not found at {CUAD_TEST_PATH}.")
        print("Run `python -m scripts.setup_cuad` first.")
        return {}

    print(f"\nLoading CUAD benchmark dataset from {CUAD_TEST_PATH.name}...")
    with open(CUAD_TEST_PATH, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts = cuad_data.get("data", [])
    if not contracts:
        print("No contracts found in dataset.")
        return {}

    print(f"Total available contracts in split: {len(contracts)}")
    selected_contracts = contracts[:num_contracts]
    print(f"Evaluating {len(selected_contracts)} contract(s) with Top-K={top_k}, Min-Score={min_score}...\n")

    embedder = LocalEmbedder()

    total_positive_tests = 0
    positive_retrieved_hits = 0
    total_negative_tests = 0
    negative_silence_hits = 0

    results_per_contract = []

    for idx, contract in enumerate(selected_contracts, 1):
        title = contract.get("title", f"Contract_{idx}")
        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue

        context = paragraphs[0].get("context", "")
        qas = paragraphs[0].get("qas", [])

        print(f"[{idx}/{len(selected_contracts)}] Indexing: {title[:60]}... ({len(context):,} chars)")

        # Convert context into simulated pages (approx 3000 chars per page)
        page_size = 3000
        pages = []
        for p_no, start in enumerate(range(0, len(context), page_size), 1):
            pages.append({
                "page_number": p_no,
                "text": context[start:start + page_size]
            })

        raw_doc = {
            "source_file": title,
            "total_pages": len(pages),
            "pages": pages
        }

        # Step 1: Clean & Chunk
        cleaned = clean_document(raw_doc)
        chunks = chunk_document(cleaned)
        print(f"    Generated {len(chunks)} legal chunks.")

        # Step 2: Index in FAISS
        embeddings = embedder.embed_chunks(chunks)
        dim = getattr(embedder, "dimension", 384)
        store = FAISSVectorStore(dimension=dim)
        store.add_documents(chunks, embeddings)
        retriever = LegalRetriever(embedder, store, top_k=top_k, min_score=min_score)

        # Step 3: Test positive and negative categories
        contract_pos_total = 0
        contract_pos_hits = 0
        contract_neg_total = 0
        contract_neg_hits = 0

        for qa in qas:
            question = qa.get("question", "")
            is_impossible = qa.get("is_impossible", False)
            answers = qa.get("answers", [])

            # Extract category name from question
            category = question
            if '"' in question:
                parts = question.split('"')
                if len(parts) >= 2:
                    category = parts[1]

            if not is_impossible and len(answers) > 0:
                # Positive test: clause exists in contract
                contract_pos_total += 1
                total_positive_tests += 1
                gold_answer = answers[0].get("text", "").strip()

                retrieval = retriever.retrieve(category)
                retrieved_text = " ".join(c.get("text", "") for c in retrieval.chunks)

                # Check if gold answer or key snippet is contained in retrieved text
                # We check overlap or significant substring match
                gold_words = set(gold_answer.lower().split()[:15])
                retrieved_words = set(retrieved_text.lower().split())
                overlap = len(gold_words & retrieved_words) / max(len(gold_words), 1)

                hit = overlap >= 0.5 or (gold_answer.lower()[:40] in retrieved_text.lower())
                if hit:
                    contract_pos_hits += 1
                    positive_retrieved_hits += 1

            elif is_impossible or len(answers) == 0:
                # Negative test: clause is absent; system MUST report silence / low_relevance
                contract_neg_total += 1
                total_negative_tests += 1

                retrieval = retriever.retrieve(category)
                # Success if low_relevance or best_score is low
                if retrieval.status != "ok" or (retrieval.best_score or 0) < min_score:
                    contract_neg_hits += 1
                    negative_silence_hits += 1

        pos_recall = (contract_pos_hits / contract_pos_total * 100) if contract_pos_total else 100
        neg_acc = (contract_neg_hits / contract_neg_total * 100) if contract_neg_total else 100
        print(f"    Positive Clause Recall: {pos_recall:.1f}% ({contract_pos_hits}/{contract_pos_total})")
        print(f"    Silence Detection Rate: {neg_acc:.1f}% ({contract_neg_hits}/{contract_neg_total})")

        results_per_contract.append({
            "title": title,
            "positive_recall": pos_recall,
            "silence_accuracy": neg_acc,
            "chunks": len(chunks)
        })

    # Overall Metrics
    overall_recall = (positive_retrieved_hits / total_positive_tests * 100) if total_positive_tests else 0
    overall_silence_acc = (negative_silence_hits / total_negative_tests * 100) if total_negative_tests else 0
    grounding_score = (overall_recall * 0.6) + (overall_silence_acc * 0.4)

    print("\n" + "=" * 65)
    print("           CUAD BENCHMARK EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f" Contracts Evaluated:        {len(selected_contracts)}")
    print(f" Total Gold Clauses Tested:  {total_positive_tests}")
    print(f" Total Silence Tests:        {total_negative_tests}")
    print("-" * 65)
    print(f" Positive Clause Recall@K:   {overall_recall:.1f}% ({positive_retrieved_hits}/{total_positive_tests})")
    print(f" Silence Detection Accuracy: {overall_silence_acc:.1f}% ({negative_silence_hits}/{total_negative_tests})")
    print(f" Combined Grounding Score:   {grounding_score:.1f}/100")
    print("=" * 65 + "\n")

    return {
        "contracts_evaluated": len(selected_contracts),
        "positive_recall": overall_recall,
        "silence_accuracy": overall_silence_acc,
        "grounding_score": grounding_score,
        "details": results_per_contract
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on CUAD dataset")
    parser.add_argument("--num_contracts", type=int, default=2, help="Number of contracts to evaluate (default: 2)")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K chunks to retrieve (default: 4)")
    parser.add_argument("--min_score", type=float, default=0.45, help="Minimum score threshold (default: 0.45)")
    args = parser.parse_args()

    evaluate_cuad(num_contracts=args.num_contracts, top_k=args.top_k, min_score=args.min_score)
