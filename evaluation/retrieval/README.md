# Stage 1 — CUAD Retrieval Evaluation

Tests **only** your existing retrieval system (`LocalEmbedder` + `FAISSVectorStore` + `LegalRetriever`) against CUAD ground-truth clauses.

## Flow
```
CUAD Question
      ↓
Existing LocalEmbedder (bge-small-en-v1.5)
      ↓
Existing FAISS Index (IndexFlatIP)
      ↓
Existing LegalRetriever (Top-K Chunks + Thresholding)
      ↓
Compare with CUAD Gold Context
      ↓
Recall@1, Recall@3, Recall@5, Recall@10, MRR, Silence Detection Rate
```

## Running the Evaluation

```bash
# Smoke test (10 queries)
python evaluation/retrieval/evaluate_retrieval.py --limit 10

# Evaluate 3 contracts (~123 queries)
python evaluation/retrieval/evaluate_retrieval.py --num_contracts 3 --top_k 10

# Full run with custom cutoff
python evaluation/retrieval/evaluate_retrieval.py --num_contracts 10 --min_score 0.40
```

## Generated Outputs
Outputs are saved in `evaluation/results/retrieval/`:
- `retrieval_results.jsonl`
- `retrieval_results.csv`
- `retrieval_report.json`
- `retrieval_report.md`
