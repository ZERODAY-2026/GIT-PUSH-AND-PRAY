# Stage 3 — CUAD End-to-End Pipeline Evaluation Report

## Executive Summary

This evaluation tests your **complete production pipeline** exactly as a user experiences it: from raw contract text processing and FAISS indexing, through thresholded retrieval, to local Gemma triage response generation.

| Metric | Value | Meaning |
|---|---:|---|
| **Total Evaluated** | `5` | End-to-end queries processed |
| **Overall End-to-End Accuracy** | **`0.0%`** | Successful retrieval AND grounded answer |
| **Positive Clause Recall** | **`100.0%`** | Found gold clause in retrieved chunks |
| **Silence Detection Rate** | **`0.0%`** | Successfully suppressed false positive retrieval |

---

## Error Analysis & Failure Attribution

| Failure Category | Occurrences | Description |
|---|---:|---|
| **`EXTRACTION FAILURE`** | **`0`** | Raw text was not properly segmented into searchable blocks |
| **`RETRIEVAL FAILURE`** | **`2`** | Target clause was missed during dense vector cosine retrieval |
| **`LLM FAILURE`** | **`0`** | Correct chunk was retrieved but Gemma failed to extract the answer |
| **`GROUNDING FAILURE`** | **`3`** | Gemma introduced unverified facts not supported by the chunk |
| **`TECHNICAL FAILURE`** | **`0`** | Exception during execution (memory, timeout, index error) |
| **`UNKNOWN / REQUIRES REVIEW`** | **`0`** | Edge cases requiring human legal expert inspection |

---

## Category-wise End-to-End Performance

| Category | Total Tested | Success Rate | Retrieval Recall |
|---|---:|---:|---:|
| **Agreement Date** | 1 | **0.0%** | 100.0% |
| **Document Name** | 1 | **0.0%** | 100.0% |
| **Effective Date** | 1 | **0.0%** | 100.0% |
| **Expiration Date** | 1 | **0.0%** | 100.0% |
| **Parties** | 1 | **0.0%** | 100.0% |

---

## Sample Successful Case Study

---

## Sample Failure Case Study & Diagnosis

### Failed Category: `Document Name`
- **Contract**: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement
- **Failure Stage**: `GROUNDING FAILURE`
- **Diagnosis**: Gemma retrieved the clause but introduced extraneous unsupported statements.
- **Gold Required**: > "SUPPLY CONTRACT..."
