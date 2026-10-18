# Stage 2 — CUAD LLM Comprehension Evaluation

Evaluates your existing **Google Gemma** integration (`build_messages` + `LocalGemmaClient`) in **isolation from retrieval**.

## Flow
```
CUAD Question + CUAD Gold Context Directly
              ↓
Existing prompt_builder.build_messages
              ↓
Existing LocalGemmaClient (LM Studio / Ollama on localhost:1234)
              ↓
Generated Triage Response
              ↓
Compare with Gold Context (Token F1, Groundedness, Unsupported Claim Rate)
```

## Running the Evaluation

```bash
# Evaluate 10 examples with active local Gemma server
python evaluation/llm/evaluate_llm.py --limit 10

# Offline / CI mock mode (verifies metrics and reporting pipeline when LM Studio is off)
python evaluation/llm/evaluate_llm.py --limit 10 --mock
```

## Generated Outputs
Outputs saved in `evaluation/results/llm/`:
- `llm_results.jsonl`
- `llm_results.csv`
- `llm_report.json`
- `llm_report.md`
