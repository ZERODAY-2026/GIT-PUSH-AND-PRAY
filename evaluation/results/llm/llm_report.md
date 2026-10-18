# Stage 2 — CUAD LLM Comprehension Evaluation Report

## Executive Summary

This evaluation tests **Google Gemma** (via `LocalGemmaClient` and `prompt_builder.build_messages`) in **isolation from retrieval** by supplying verified CUAD gold contract evidence directly.

| Metric | Value | Meaning |
|---|---:|---|
| **Model Tested** | `local-model` | On-device inference model |
| **Total Examples Evaluated** | `10` | Positive CUAD clause questions |
| **Mean Token F1** | **`8.8%`** | Word overlap between Gemma answer and gold clause |
| **Key-Term Recall** | **`96.6%`** | Preservation of critical operative terms |
| **Groundedness Score** | **`9.6%`** | Claims directly substantiated by provided text |
| **Unsupported Claim Rate** | **`90.4%`** | Extraneous facts introduced (hallucinations) |

---

## Category-wise LLM Performance

| Category | Examples | Mean F1 | Groundedness | Unsupported Rate |
|---|---:|---:|---:|---:|
| **Agreement Date** | 1 | 3.5% | 0.0% | 100.0% |
| **Document Name** | 2 | 2.2% | 0.0% | 100.0% |
| **Effective Date** | 1 | 17.9% | 27.3% | 72.7% |
| **Expiration Date** | 1 | 7.5% | 9.1% | 90.9% |
| **Governing Law** | 1 | 14.0% | 20.0% | 80.0% |
| **Insurance** | 1 | 9.2% | 11.1% | 88.9% |
| **Parties** | 2 | 1.1% | 0.0% | 100.0% |
| **Warranty Duration** | 1 | 29.1% | 28.6% | 71.4% |

---

## Sample Grounded Generations

### Category: `Document Name`
- **Gold Evidence**: > "SUPPLY CONTRACT..."
- **Gemma Output (Snippet)**:

```markdown
### 1. Explicit Document Statements
The document explicitly identifies itself and provides metadata fields related to its naming convention:

*   **Contract Type:** The overall agreement is titled a "SUPPLY CONTRACT" [SOURCE 1].
*   **Naming Metadata:** The excerpts show structural labels for the contract's name, specifically listing `Document Name | Clause Document Name` [SOURCE 1].

The document...
```
- **Scores**: F1 = `0.0129`, Groundedness = `0.0`, Latency = `48.94s`

### Category: `Parties`
- **Gold Evidence**: > "The seller:..."
- **Gemma Output (Snippet)**:

```markdown
### 1. Explicit Document Statements
The document explicitly identifies one role within the parties section: **"The seller:"** [SOURCE 1].

No other party names, roles, or details regarding who signed the contract are provided in these excerpts.

### 2. Potential Interpretations & Practical Concerns
The provided excerpt is highly incomplete and only lists a single role ("The seller:") [SOURCE 1]. I...
```
- **Scores**: F1 = `0.008`, Groundedness = `0.0`, Latency = `46.62s`
