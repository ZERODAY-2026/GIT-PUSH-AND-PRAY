# Developer & AI Assistant Handover Guide

> **Project:** `localize-legal-RAG` (Privacy-first, on-device legal document triage agent)  
> **Target Audience:** Teammate building the **Querying / Retrieval / Gemma Reasoning layer**, and any AI assistant pair-programming with them.  
> **Status:** Modules 1–5 (Ingestion, Cleaning, Chunking, Local Embeddings, and Persistent Vector Store) are completed, tested, and working.

---

## 1. Quick Summary: What Is Built vs. What You Are Building

### ✅ Completed & Ready to Use (Modules 1–5):
1. **Module 1 (PDF Ingestion):** Extracts text page-by-page from local PDFs (`src/ingestion/pdf_extractor.py`).
2. **Module 2 (Cleaning):** Cleans text conservatively without losing legal meaning, negations, numbers, or amounts; strips confirmed repeated headers/footers (`src/processing/cleaner.py`).
3. **Module 3 (Semantic Chunking):** Splits text into legal clauses/sections, preserving exact page numbers, section headers, and clause tags (`src/processing/chunker.py`).
4. **Module 4 (Local Embeddings):** Generates 384-dimensional dense vectors using `BAAI/bge-small-en-v1.5` on CPU (`src/embedding/embedder.py`).
5. **Module 5 (FAISS Vector Store):** Stores normalized vectors in FAISS `IndexFlatIP` and decouples metadata into JSON (`src/vector_store/faiss_store.py`).

### 🎯 Your Task (Modules 6+ — Retrieval, Prompting & Gemma):
1. **Query Processing & Retrieval:** Accept user questions in natural language, embed the question, retrieve the top-$k$ most relevant chunks from the FAISS store, and filter/rerank if needed.
2. **Context Assembly & Grounding:** Format retrieved chunks into a prompt with exact source citations (`page_start`, `page_end`, `section`, `clause`).
3. **Local Gemma Inference:** Run on-device Gemma (e.g. Gemma 3n or 4 edge models via llama-cpp-python, Ollama, or Hugging Face transformers).
4. **Calibrated Output & Guardrails:** Generate plain-language explanations with explicit page citations, communicate uncertainty for ambiguous/missing clauses, and append legal disclaimer (never present as definitive legal advice).

---

## 2. Environment Setup (First 2 Minutes)

### Recommended Python Version
* **Use CPython 3.11** (or 3.10–3.12). Avoid Python 3.14 on MSYS2, as `faiss-cpu` and `pymupdf` have pre-built wheels for 3.10–3.12/3.13 only.

### Setup Instructions
```powershell
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # On Windows (or source .venv/bin/activate on Linux/Mac)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify that all 59 tests pass
pytest tests/ -v
```

---

## 3. Directory Layout & Key File Locations

```
localize-legal-RAG/
│
├── data/
│   ├── input/                  ← Place raw legal PDFs here (e.g. trial.pdf, rental_agreement.pdf)
│   ├── extracted/              ← Module 1 outputs: <doc>.json & <doc>.txt
│   ├── processed/              ← Module 2 & 3 outputs: <doc>_cleaned.json & <doc>_chunks.json
│   └── vector_store/           ← Module 5 FAISS index and metadata:
│       ├── index.faiss         ← Binary FAISS vector index
│       ├── mapping.json        ← Vector integer ID -> chunk_id
│       └── chunks.json         ← chunk_id -> chunk metadata & text
│
├── src/
│   ├── ingestion/
│   │   └── pdf_extractor.py    ← PDF parsing logic
│   ├── processing/
│   │   ├── cleaner.py          ← Legal text cleaning & header/footer removal
│   │   ├── chunker.py          ← Semantic chunking & deterministic IDs
│   │   └── pipeline.py         ← Runs Steps 2-5 end-to-end
│   ├── embedding/
│   │   └── embedder.py         ← LocalEmbedder (BAAI/bge-small-en-v1.5)
│   └── vector_store/
│       └── faiss_store.py      ← FAISSVectorStore class (IndexFlatIP)
│
├── tests/                      ← 59 passing pytest unit and integration tests
├── requirements.txt            ← pip dependencies
├── README.md                   ← Architecture and pipeline guide
└── HANDOVER_GUIDE.md           ← This file
```

---

## 4. Vector Store & Database Details

### Model & Vector Specs
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (loaded via `sentence-transformers`).
* **Vector Dimensionality:** `384` float32 dimensions.
* **Vector Normalization:** Vectors are **L2-normalized** (`np.linalg.norm == 1.0`).
* **FAISS Index Type:** `faiss.IndexFlatIP(384)` (Inner Product on unit-normalized vectors equals **exact Cosine Similarity**).
* **Score Interpretation:**
  * Range: `[-1.0, 1.0]`.
  * Typical relevant match: `0.65 – 0.85+`.
  * Irrelevant match: `< 0.40`.

### Storage Files in `data/vector_store/`
The vector store decouples heavy vectors from document metadata so multi-megabyte floats are not duplicated in JSON:
1. **`index.faiss`**: The binary FAISS index file.
2. **`mapping.json`**: Maps FAISS internal row index to string chunk ID:
   ```json
   {
     "0": "trial_p001_c001",
     "1": "trial_p002_c002",
     "2": "trial_p003_c003"
   }
   ```
3. **`chunks.json`**: Maps `chunk_id` directly to full chunk text and provenance:
   ```json
   {
     "trial_p001_c001": {
       "chunk_id": "trial_p001_c001",
       "document_id": "trial",
       "source_file": "trial.pdf",
       "page_start": 1,
       "page_end": 2,
       "section": "SECTION 1: PARTIES",
       "clause": "1.1",
       "text": "Landlord agrees to lease the premises to Tenant...",
       "char_count": 450,
       "token_count": 85
     }
   }
   ```

---

## 5. Exact JSON Schemas for Each Pipeline Stage

### Schema 1: Extracted PDF (`data/extracted/<name>.json`)
Produced by `src.ingestion.pdf_extractor`:
```json
{
  "source_file": "trial.pdf",
  "source_path": "C:\\path\\to\\trial.pdf",
  "total_pages": 11,
  "total_chars": 26775,
  "scanned_pages": 0,
  "extraction_error": null,
  "pages": [
    {
      "page_number": 1,
      "char_count": 2285,
      "needs_ocr": false,
      "text": "Raw extracted text from page 1..."
    }
  ]
}
```

### Schema 2: Cleaned Document (`data/processed/<name>_cleaned.json`)
Produced by `src.processing.cleaner`:
```json
{
  "document_id": "trial",
  "source_file": "trial.pdf",
  "source_path": "C:\\path\\to\\trial.pdf",
  "total_pages": 11,
  "total_chars_original": 26775,
  "total_chars_cleaned": 24952,
  "pages": [
    {
      "page_number": 1,
      "char_count": 2150,
      "text": "Cleaned text with boilerplate headers/footers removed..."
    }
  ]
}
```

### Schema 3: Chunks (`data/processed/<name>_chunks.json`)
Produced by `src.processing.chunker`:
```json
[
  {
    "chunk_id": "trial_p001_c001",
    "document_id": "trial",
    "source_file": "trial.pdf",
    "page_start": 1,
    "page_end": 2,
    "section": "PS1 — DataMind: Talk to Your Data",
    "clause": "M1",
    "text": "Chunk text content...",
    "char_count": 3310,
    "token_count": 611
  }
]
```
*Note on `section` & `clause`: If not present in the document text, they are `null` (never invented).*

---

## 6. How to Query the Vector Store (Code Examples for You)

You do **not** need to touch FAISS C++ bindings or calculate vector dots manually. The `FAISSVectorStore` and `LocalEmbedder` classes handle everything.

### Example: Semantic Search in Python
```python
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore

# 1. Load the pre-indexed vector store from data/vector_store
store = FAISSVectorStore.load("data/vector_store")

# 2. Initialize the local embedder (uses BAAI/bge-small-en-v1.5)
embedder = LocalEmbedder()

# 3. User question
question = "What happens if rent is paid late?"

# 4. Embed question
query_vector = embedder.embed_texts([question])  # shape: (1, 384)

# 5. Search top-k most relevant chunks
results = store.search(query_vector, top_k=3)

# 6. Inspect retrieved results
for i, chunk in enumerate(results, 1):
    print(f"--- Result #{i} (Score: {chunk['score']:.4f}) ---")
    print(f"Chunk ID:   {chunk['chunk_id']}")
    print(f"Pages:      {chunk['page_start']} to {chunk['page_end']}")
    print(f"Section:    {chunk['section']}")
    print(f"Clause:     {chunk['clause']}")
    print(f"Passage:    {chunk['text'][:250]}...\n")
```

### Output Format of `store.search(...)`
Returns a `list[dict]`, where each element contains:
* `chunk_id` (str): e.g. `"trial_p001_c001"`
* `score` (float): Cosine similarity score (higher = closer, typically 0.65 to 0.90)
* `page_start` (int): 1-based starting page
* `page_end` (int): 1-based ending page
* `section` (str | None): Detected section title
* `clause` (str | None): Detected clause identifier
* `text` (str): Plain-text body of the chunk
* `document_id` (str): e.g. `"trial"`
* `source_file` (str): e.g. `"trial.pdf"`

---

## 7. How to Re-Run Ingestion on a New PDF

If you receive a new legal document (e.g. `rental_agreement.pdf`):

```powershell
# Step 1: Extract PDF -> data/extracted/rental_agreement.json
python -m src.ingestion.pdf_extractor data/input/rental_agreement.pdf

# Step 2: Clean, Chunk, Embed, and Index into FAISS
python -m src.processing.pipeline data/extracted/rental_agreement.json
```

---

## 8. What to Build Next: Architecture Recommendation

Here is the recommended blueprint for your retrieval and Gemma reasoning module:

```
User Question
      │
      ▼
1. Query Embedding (LocalEmbedder.embed_texts)
      │
      ▼
2. Vector Retrieval (FAISSVectorStore.search(top_k=3..5))
      │
      ▼
3. Threshold Filtering (Filter chunks with score < 0.50; flag uncertainty if no good chunks)
      │
      ▼
4. Prompt Assembly (Build prompt with retrieved chunks + page citations + legal boundary instructions)
      │
      ▼
5. Local Gemma 3n/4 Inference (On-device LLM)
      │
      ▼
6. Grounded Output Formatting:
   - Plain-language explanation
   - Cited passages: [Page X, Clause Y]
   - Identified obligations / penalties / deadlines
   - Uncertainty flag (if question is ambiguous or not addressed in the text)
   - Mandatory Disclaimer: "Not legal advice; preliminary triage only."
```

### Recommended Gemma System Prompt Guardrail
> *"You are a private, on-device legal document assistant. Answer ONLY based on the provided document excerpts. Always cite the exact page number and clause. If the document does not contain enough information to answer with confidence, state that clearly and recommend consulting a qualified legal professional. Never present your answer as definitive legal advice."*

---

## 9. Privacy & Network Guardrails (Strict Requirement)

* **Zero External APIs:** Do NOT use OpenAI, Gemini Cloud API, Anthropic, or any remote REST endpoint.
* **Local Weights Only:** The embedding model weights (`bge-small-en-v1.5`) are already downloaded to `~/.cache/huggingface/hub`. The pipeline functions 100% offline.
* **Document Data Privacy:** Never print complete legal contracts to stdout/logs. Log only counts, page numbers, and execution times.

---

## 10. Direct Instructions for Your AI Coding Assistant

If you are using Claude, ChatGPT, Gemini, Copilot, or Cursor to help you build the next module, copy-paste this prompt into your AI chat:

```text
I am building the retrieval and question-answering module for an on-device legal document assistant (localize-legal-RAG).

The retrieval layer is already implemented in Python:
- FAISS vector store is located in 'data/vector_store/' (index.faiss, mapping.json, chunks.json).
- Embedding model is 'BAAI/bge-small-en-v1.5' (384 dimensions, normalized, loaded via src.embedding.embedder.LocalEmbedder).
- FAISS index uses IndexFlatIP (cosine similarity).
- To load and search:
    from src.vector_store.faiss_store import FAISSVectorStore
    from src.embedding.embedder import LocalEmbedder
    store = FAISSVectorStore.load("data/vector_store")
    embedder = LocalEmbedder()
    results = store.search(embedder.embed_texts([query]), top_k=3)
- Each result dict contains: 'chunk_id', 'score', 'page_start', 'page_end', 'section', 'clause', 'text', 'source_file'.

My task now:
1. Build a retriever module that accepts user questions, retrieves top chunks, and filters low-confidence matches.
2. Build an augmented prompt containing the retrieved context with explicit page/section citations.
3. Integrate an on-device Gemma model (e.g. llama-cpp-python or transformers) to produce grounded, plain-language answers with uncertainty calibration and legal disclaimers.
4. Privacy constraint: All processing must remain completely local and on-device. No external APIs.
```
