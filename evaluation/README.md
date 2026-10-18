# CUAD Benchmark Evaluation Framework

A completely self-contained, zero-modification evaluation harness for testing the **Privacy-First Legal Document AI Triage System** against the expert-annotated **Contract Understanding Atticus Dataset (CUAD)**.

---

## Architecture Overview

```
MY EXISTING SYSTEM (Untouched)
        ↑
        | (Calls existing components as an external tester)
   EVALUATION HARNESS
        |
   CUAD GROUND TRUTH (510 Contracts, 41 Categories, 13,000+ Human Annotations)
```

### The 3 Independent Evaluation Stages

| Stage | What Is Evaluated | Input | Output / Metrics |
|---|---|---|---|
| **Stage 1: Retrieval** | `LocalEmbedder` + `FAISSVectorStore` + `LegalRetriever` | CUAD Question | Recall@1, Recall@3, Recall@5, Recall@10, MRR, Silence Detection Rate |
| **Stage 2: LLM Comprehension** | `build_messages` + `LocalGemmaClient` | CUAD Question + CUAD Gold Context directly | Token F1, Exact Match, Groundedness Score, Unsupported Claim Rate |
| **Stage 3: End-to-End Pipeline** | Raw Contract Text → Clean → Chunk → Embed → FAISS → Retrieve → Gemma | Raw Contract Text + CUAD Question | Joint Accuracy, Category Performance, Error Failure Attribution |

---

## Directory Structure

```
evaluation/
│
├── cuad/
│   ├── prepare_cuad.py               # Parse and normalize raw CUAD into standardized JSON
│   ├── data/
│   │   └── cuad_processed.json       # Normalized benchmark dataset
│   └── README.md
│
├── retrieval/
│   ├── evaluate_retrieval.py         # Stage 1: Dense retrieval evaluation
│   ├── matching.py                   # Normalized string & token overlap evidence matcher
│   ├── metrics.py                    # Recall@K and MRR computation
│   └── README.md
│
├── llm/
│   ├── evaluate_llm.py               # Stage 2: Isolated Gemma comprehension evaluation
│   ├── metrics.py                    # Token F1, Groundedness, Hallucination calculation
│   └── README.md
│
├── end_to_end/
│   ├── evaluate_pipeline.py          # Stage 3: Full user pipeline & error attribution
│   ├── metrics.py                    # Joint pipeline metrics & failure taxonomy
│   └── README.md
│
├── visualize.py                      # Publication-quality chart generator (matplotlib)
├── generate_final_report.py          # Consolidated whitepaper synthesizer
│
├── results/
│   ├── retrieval/                    # Stage 1 jsonl, csv, json, md
│   ├── llm/                          # Stage 2 jsonl, csv, json, md
│   ├── end_to_end/                   # Stage 3 jsonl, csv, json, md
│   ├── plots/                        # PNG visual graphs
│   └── final_CUAD_evaluation_report.md # Final unified evaluation report
│
└── README.md
```

---

## Step-by-Step Execution Guide

### Step 1: Prepare the CUAD Dataset

Normalizes the official CUAD annotations into standardized evaluation examples:

```bash
# Normalize contracts from CUAD split
python evaluation/cuad/prepare_cuad.py

# Or limit to first 10 contracts for fast iteration
python evaluation/cuad/prepare_cuad.py --max_contracts 10
```

### Step 2: Run Stage 1 — Retrieval Evaluation

Tests whether the dense vector index surfaces the correct legal clause in the top chunks:

```bash
# Fast smoke-test (10 queries)
python evaluation/retrieval/evaluate_retrieval.py --limit 10

# Standard benchmark (3 contracts, ~123 category queries)
python evaluation/retrieval/evaluate_retrieval.py --num_contracts 3 --top_k 10 --min_score 0.40
```
**Outputs Generated**:
- `evaluation/results/retrieval/retrieval_results.jsonl`
- `evaluation/results/retrieval/retrieval_results.csv`
- `evaluation/results/retrieval/retrieval_report.json`
- `evaluation/results/retrieval/retrieval_report.md`

### Step 3: Run Stage 2 — LLM Comprehension Evaluation

Tests Gemma by feeding it the verified CUAD gold clause context directly:

```bash
# With active local Gemma server in LM Studio / Ollama
python evaluation/llm/evaluate_llm.py --limit 10

# Offline test mode (verifies metrics and reporting pipeline when LM Studio is off)
python evaluation/llm/evaluate_llm.py --limit 10 --mock
```
**Outputs Generated**:
- `evaluation/results/llm/llm_results.jsonl`
- `evaluation/results/llm/llm_results.csv`
- `evaluation/results/llm/llm_report.json`
- `evaluation/results/llm/llm_report.md`

### Step 4: Run Stage 3 — End-to-End Pipeline Evaluation

Tests the complete system from raw document ingestion to final answer:

```bash
# Fast smoke-test (10 queries)
python evaluation/end_to_end/evaluate_pipeline.py --limit 10 --mock

# Full live test
python evaluation/end_to_end/evaluate_pipeline.py --num_contracts 2 --limit 20
```
**Outputs Generated**:
- `evaluation/results/end_to_end/end_to_end_results.jsonl`
- `evaluation/results/end_to_end/end_to_end_results.csv`
- `evaluation/results/end_to_end/end_to_end_report.json`
- `evaluation/results/end_to_end/end_to_end_report.md`

### Step 5: Generate Visual Plots and Final Report

```bash
# Generate charts in evaluation/results/plots/
python evaluation/visualize.py

# Generate final comprehensive whitepaper
python evaluation/generate_final_report.py
```

---

## What the Metrics Mean

* **Recall@1**: Did the first retrieved chunk contain the CUAD gold clause?
* **Recall@5**: Was the gold clause present in the top 5 chunks passed to the LLM?
* **MRR (Mean Reciprocal Rank)**: How close to rank 1 the first relevant chunk landed on average ($1.0$ is perfect).
* **Silence Detection Accuracy**: When a contract does NOT contain a clause, did the system correctly suppress retrieval instead of hallucinating irrelevant text?
* **Token F1**: Precision and recall harmonic mean between Gemma's answer and the gold clause.
* **Groundedness Score**: Percentage of factual claims made by Gemma that are directly supported by the document text.
* **Unsupported Claim Rate**: Percentage of statements introduced by Gemma that are NOT found in the contract text (hallucinations).
* **Failure Attribution**: Categorizes errors into Extraction, Retrieval, LLM, Grounding, or Technical failure.
