# CUAD Benchmark Data Preparation

This module standardizes the **Contract Understanding Atticus Dataset (CUAD)** into structured benchmark records for evaluating the existing legal document AI system without modifying any production code.

## What It Does
1. Parses official CUAD annotations (`test.json` or `CUADv1.json`).
2. Extracts and pairs:
   - Contract full text (`context`)
   - 41 standard legal clause categories
   - Verbatim human-annotated gold answers (`gold_spans`)
   - Silence flags (`is_impossible: True` when contract contains no such provision)
3. Outputs clean JSON to `evaluation/cuad/data/cuad_processed.json`.

## Usage

```bash
# Prepare all contracts from split
python evaluation/cuad/prepare_cuad.py

# Or limit to first 10 contracts for rapid development
python evaluation/cuad/prepare_cuad.py --max_contracts 10
```
