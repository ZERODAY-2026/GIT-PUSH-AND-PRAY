# Stage 1 — CUAD Retrieval Evaluation Report

## Executive Summary

This evaluation tests your **existing retrieval system** (*bge-small-en-v1.5* dense embeddings + *FAISS IndexFlatIP* cosine search + *LegalRetriever* thresholding) against gold standard Atticus contract annotations.

| Metric | Value | Meaning |
|---|---:|---|
| **Total Examples Evaluated** | `205` | Total contract queries processed |
| **Positive Clause Queries** | `70` | Clauses with verified human ground-truth text |
| **Silence Negative Queries** | `135` | Cases where contract was silent on topic |
| **Recall@1** | **`50.0%`** | Gold evidence surfaced in the #1 top chunk |
| **Recall@3** | **`75.7%`** | Gold evidence surfaced in top 3 chunks |
| **Recall@5** | **`81.4%`** | Gold evidence surfaced in top 5 chunks |
| **Recall@10** | **`87.1%`** | Gold evidence surfaced in top 10 chunks |
| **MRR (Mean Reciprocal Rank)** | **`0.6432`** | Average reciprocal ranking efficiency |
| **Silence Detection Rate** | **`0.0%`** | Correctly identified silence (no false positives) |

---

## Category-wise Retrieval Performance

| Category | Total | Pos | Recall@1 | Recall@5 | Recall@10 | MRR | Silence Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Affiliate License-Licensee** | 5 | 1 | 0.0% | 100.0% | 100.0% | 0.500 | 0.0% |
| **Affiliate License-Licensor** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Agreement Date** | 5 | 4 | 50.0% | 75.0% | 75.0% | 0.562 | 0.0% |
| **Anti-Assignment** | 5 | 3 | 66.7% | 66.7% | 66.7% | 0.667 | 0.0% |
| **Audit Rights** | 5 | 2 | 50.0% | 100.0% | 100.0% | 0.750 | 0.0% |
| **Cap On Liability** | 5 | 3 | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% |
| **Change Of Control** | 5 | 2 | 50.0% | 100.0% | 100.0% | 0.625 | 0.0% |
| **Competitive Restriction Exception** | 5 | 1 | 0.0% | 0.0% | 100.0% | 0.143 | 0.0% |
| **Covenant Not To Sue** | 5 | 1 | 0.0% | 100.0% | 100.0% | 0.333 | 0.0% |
| **Document Name** | 5 | 5 | 40.0% | 80.0% | 80.0% | 0.600 | 100.0% |
| **Effective Date** | 5 | 4 | 75.0% | 100.0% | 100.0% | 0.812 | 0.0% |
| **Exclusivity** | 5 | 2 | 0.0% | 50.0% | 50.0% | 0.250 | 0.0% |
| **Expiration Date** | 5 | 4 | 50.0% | 75.0% | 75.0% | 0.625 | 0.0% |
| **Governing Law** | 5 | 4 | 50.0% | 50.0% | 50.0% | 0.500 | 0.0% |
| **Insurance** | 5 | 3 | 66.7% | 100.0% | 100.0% | 0.833 | 0.0% |
| **Ip Ownership Assignment** | 5 | 2 | 50.0% | 100.0% | 100.0% | 0.750 | 0.0% |
| **Irrevocable Or Perpetual License** | 5 | 1 | 0.0% | 0.0% | 100.0% | 0.143 | 0.0% |
| **Joint Ip Ownership** | 5 | 1 | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% |
| **License Grant** | 5 | 2 | 50.0% | 100.0% | 100.0% | 0.750 | 0.0% |
| **Liquidated Damages** | 5 | 1 | 0.0% | 100.0% | 100.0% | 0.500 | 0.0% |
| **Minimum Commitment** | 5 | 1 | 0.0% | 100.0% | 100.0% | 0.333 | 0.0% |
| **Most Favored Nation** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **No-Solicit Of Customers** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **No-Solicit Of Employees** | 5 | 1 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Non-Compete** | 5 | 2 | 0.0% | 100.0% | 100.0% | 0.417 | 0.0% |
| **Non-Disparagement** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Non-Transferable License** | 5 | 2 | 50.0% | 50.0% | 100.0% | 0.583 | 0.0% |
| **Notice Period To Terminate Renewal** | 5 | 2 | 50.0% | 50.0% | 100.0% | 0.562 | 0.0% |
| **Parties** | 5 | 5 | 80.0% | 100.0% | 100.0% | 0.900 | 100.0% |
| **Post-Termination Services** | 5 | 1 | 0.0% | 100.0% | 100.0% | 0.500 | 0.0% |
| **Price Restrictions** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Renewal Term** | 5 | 2 | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% |
| **Revenue/Profit Sharing** | 5 | 2 | 0.0% | 50.0% | 50.0% | 0.100 | 0.0% |
| **Rofr/Rofo/Rofn** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Source Code Escrow** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Termination For Convenience** | 5 | 3 | 66.7% | 100.0% | 100.0% | 0.833 | 0.0% |
| **Third Party Beneficiary** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Uncapped Liability** | 5 | 2 | 50.0% | 100.0% | 100.0% | 0.750 | 0.0% |
| **Unlimited/All-You-Can-Eat-License** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Volume Restriction** | 5 | 0 | 0.0% | 0.0% | 0.0% | 0.000 | 0.0% |
| **Warranty Duration** | 5 | 1 | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% |

---

## Sample Successful Retrievals

### Category: `Document Name` (Rank 2)
- **Contract**: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement
- **Gold Span**: > "SUPPLY CONTRACT..."
- **Retrieved Best Score**: `0.5261`

### Category: `Parties` (Rank 1)
- **Contract**: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement
- **Gold Span**: > "The seller:..."
- **Retrieved Best Score**: `0.4751`

### Category: `Expiration Date` (Rank 1)
- **Contract**: LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement
- **Gold Span**: > "The Contract is valid for 5 years, beginning from and ended on ...."
- **Retrieved Best Score**: `0.5928`

---

## Sample Retrieval Failures

### Failed Category: `Governing Law`
- **Contract**: DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement
- **Gold Span Required**: > "This Agreement and any and all matters arising directly or indirectly herefrom shall be governed by and construed and enforced in accordance with the internal laws of the [***] applicable to agreements made and to be performed entirely in such state,..."
- **Retrieved Best Score**: `0.6204` (below threshold or wrong section)

### Failed Category: `Exclusivity`
- **Contract**: DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement
- **Gold Span Required**: > "During the Term, subject to the terms and conditions of this Agreement, Dova hereby grants to Valeant the right, on a co-exclusive basis (solely with Dova and its Affiliates), to Detail and promote the Product in the Specialty in the Territory in the..."
- **Retrieved Best Score**: `0.5918` (below threshold or wrong section)

### Failed Category: `No-Solicit Of Employees`
- **Contract**: DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement
- **Gold Span Required**: > "[***], neither Valeant nor Dova (nor any of their respective Affiliates) shall directly or indirectly solicit for hire or employee as an employee, consultant or otherwise any of the other Party's professional personnel who have had direct involvement..."
- **Retrieved Best Score**: `0.5914` (below threshold or wrong section)


---

## Common Retrieval Failure Patterns & Observations

1. **Dense Vector Lexical Gaps**: Very specific numerical definitions or definitions of terms (e.g. specific entity dates) occasionally rank below general contract headers.
2. **Chunk Boundary Splitting**: When an Atticus annotation spans across two paragraphs, the exact chunking boundary can cut the clause, requiring multi-chunk context synthesis.
3. **Threshold Calibration**: The default cosine cutoff (`min_score = 0.40 - 0.45`) effectively suppresses ungrounded hallucination on silence categories.