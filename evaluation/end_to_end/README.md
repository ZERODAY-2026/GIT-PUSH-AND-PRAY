# Stage 3 — CUAD End-to-End Pipeline Evaluation

Evaluates the **entire legal document AI system** exactly as an end user interacts with it.

## Architecture
```
CUAD Contract Text + CUAD Question
              ↓
Existing Ingestion / Simulation
              ↓
Existing Cleaner (clean_document)
              ↓
Existing Chunker (chunk_document)
              ↓
Existing LocalEmbedder (bge-small-en-v1.5)
              ↓
Existing FAISS Store (IndexFlatIP)
              ↓
Existing LegalRetriever (Top-K Chunks + Thresholding)
              ↓
Top Retrieved Chunks ONLY (No gold leakage!)
              ↓
Existing LocalGemmaClient (LM Studio / Ollama)
              ↓
Final Triage Answer
              ↓
Evaluation & Error Attribution (Extraction, Retrieval, LLM, Grounding)
```

## Running the Evaluation

```bash
# Smoke test (10 queries with mock Gemma if offline)
python evaluation/end_to_end/evaluate_pipeline.py --limit 10 --mock

# Live test with active local Gemma server
python evaluation/end_to_end/evaluate_pipeline.py --num_contracts 2 --limit 20
```

## Generated Outputs
Outputs saved in `evaluation/results/end_to_end/`:
- `end_to_end_results.jsonl`
- `end_to_end_results.csv`
- `end_to_end_report.json`
- `end_to_end_report.md`
