# localize-legal-RAG

> **Privacy-first, fully on-device AI agent for analyzing legal documents.**
>
> All processing happens on your machine. Sensitive legal documents, extracted text, chunks, embeddings, and indices never leave your device.

---

## Project Overview

This project implements an on-device Retrieval-Augmented Generation (RAG) system specifically designed for legal documents (rental agreements, employment contracts, service agreements, and legal notices).

The system architecture cleanly decouples document retrieval from language reasoning:

* **Embedding Model (`BAAI/bge-small-en-v1.5`):** Specializes in *finding* the most relevant legal clauses and passages from the document vector index.
* **Gemma Model (Future stage):** Specializes in *reading, synthesizing, and explaining* those retrieved passages in plain language, citing exact source pages and communicating calibrated uncertainty without offering legal advice.

### Current Pipeline Status

```
PDF (e.g. data/input/rental_agreement.pdf)
 │
 ▼ [MODULE 1: INGESTION]
Local PDF Text Extraction (PyMuPDF / local OCR fallback)
 │
 ▼ (data/extracted/rental_agreement.json)
 │
 ▼ [MODULE 2: PROCESSING]
Conservative Legal Text Cleaning & Header/Footer Detection
 │
 ▼ (data/processed/rental_agreement_cleaned.json)
 │
 ▼ [MODULE 3: CHUNKING]
Legal-Aware Semantic Chunking (Section -> Clause -> Sentence)
 │
 ▼ (data/processed/rental_agreement_chunks.json)
 │
 ▼ [MODULE 4: EMBEDDING]
Local Sentence-Transformers Embeddings (bge-small-en-v1.5, normalized)
 │
 ▼ [MODULE 5: VECTOR STORE]
Local FAISS IndexFlatIP + Decoupled Provenance Metadata
   ├── data/vector_store/index.faiss
   ├── data/vector_store/mapping.json
   └── data/vector_store/chunks.json
 │
 ▼ [FUTURE MILESTONE]
Local Gemma Query Reasoning + Citation + Calibrated Uncertainty
```

---

## Project Structure

```
localize-legal-RAG/
│
├── data/
│   ├── input/                  ← input PDFs (git-ignored)
│   ├── extracted/              ← Module 1 extracted text & JSON (git-ignored)
│   ├── processed/              ← Module 2 cleaned JSON & Module 3 chunks (git-ignored)
│   └── vector_store/           ← Module 5 FAISS index & metadata mapping (git-ignored)
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_extractor.py    ← Module 1: Local PDF text extraction & CLI
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py          ← Module 2: Conservative legal cleaning & header/footer removal
│   │   ├── chunker.py          ← Module 3: Legal-aware semantic chunking & provenance
│   │   └── pipeline.py         ← End-to-end processing pipeline orchestrator & CLI
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py         ← Module 4: Local sentence-transformers embeddings
│   │
│   └── vector_store/
│       ├── __init__.py
│       └── faiss_store.py      ← Module 5: FAISS IndexFlatIP & provenance mapping
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_extractor.py   ← Ingestion test suite
│   ├── test_cleaner.py         ← Cleaning & preservation test suite
│   ├── test_chunker.py         ← Semantic chunking test suite
│   ├── test_embedder.py        ← Embeddings & normalization test suite
│   ├── test_vector_store.py    ← FAISS index, persistence, & retrieval test suite
│   └── test_pipeline.py        ← Pipeline integration test suite
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Dependencies

### Python Version Recommendation
Use **CPython 3.11** (or 3.10–3.12). Pre-built binary wheels for `pymupdf` and `faiss-cpu` are readily available for these versions.

### 1. Create a Virtual Environment

```powershell
C:\Users\aayus\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv311
```

### 2. Install Dependencies

```powershell
.venv311\Scripts\pip.exe install -r requirements.txt
```

Active runtime dependencies:
* `pymupdf`: On-device PDF extraction
* `sentence-transformers`: Local embedding generation
* `faiss-cpu`: Local vector similarity index
* `pytest`: Comprehensive unit & integration testing

---

## How to Run the Pipeline

### Step 1: Extract Text from Local PDF (Module 1)

Place your PDF into `data/input/` and run:

```powershell
.venv311\Scripts\python.exe -m src.ingestion.pdf_extractor data/input/trial.pdf
```

Outputs generated:
* `data/extracted/trial.txt` (plain text with page headers)
* `data/extracted/trial.json` (page-by-page structured JSON)

### Step 2: Clean, Chunk, Embed, and Index into FAISS (Modules 2–5)

Run the unified processing pipeline on the extracted JSON:

```powershell
.venv311\Scripts\python.exe -m src.processing.pipeline data/extracted/trial.json
```

Terminal output:
```text
============================================================
Processing document: trial.pdf
============================================================
Pages:               11
Cleaned pages:       11
Chunks created:      8
Embedding model:     BAAI/bge-small-en-v1.5
Embedding dimension: 384
Vectors indexed:     8

Saved:
  data\processed\trial_cleaned.json
  data\processed\trial_chunks.json
  data\vector_store\index.faiss
  data\vector_store\mapping.json
  data\vector_store\chunks.json
============================================================
```

### Optional Command-Line Flags

```powershell
# Custom chunk size and overlap
.venv311\Scripts\python.exe -m src.processing.pipeline data/extracted/trial.json --target-tokens 500 --overlap-tokens 75

# Disable automatic header/footer removal
.venv311\Scripts\python.exe -m src.processing.pipeline data/extracted/trial.json --keep-headers

# Use a different local embedding model
.venv311\Scripts\python.exe -m src.processing.pipeline data/extracted/trial.json --model all-MiniLM-L6-v2
```

---

## Architecture & Engineering Details

### 1. Conservative Legal Cleaning (`src/processing/cleaner.py`)
Legal text cleaning requires preserving legal validity over aggressive normalization:
* **Critical terms preserved:** Negations (`not`, `never`, `neither`, `nor`), modal obligations (`shall`, `must`, `may`, `will`, `prohibited`, `unless`, `except`, `without`), and condition clauses are strictly preserved.
* **Numbers & currency preserved:** Monetary figures (`$2,500`, `₹50,000`), percentages (`5%`), dates, deadlines, and clause numbers (`1.1`, `(a)`) are kept intact.
* **Safe line-wrap repair:** Repaired only when a line does not end with terminal punctuation and the following line is not a heading or new clause.
* **Boilerplate header/footer detection:** Multi-page documents are analyzed to identify lines that repeat consistently across pages (>= 50% frequency). Lines matching confirmed boilerplate or pagination patterns are removed, while genuine legal clauses are protected.
* **Integrity validation:** Emits a warning if cleaning removes more than 35% of characters.

### 2. Legal-Aware Semantic Chunking (`src/processing/chunker.py`)
Naive fixed-character chunking splits legal clauses across arbitrary boundaries, corrupting context. Our chunker follows a semantic hierarchy:
1. **Document**
2. **Section** (e.g. `SECTION 7 — TERMINATION`, `ARTICLE IV: RENT`)
3. **Clause** (e.g. `7.1`, `7.2`, `Clause 4`)
4. **Sub-clause** (e.g. `(a)`, `(b)`)
5. **Paragraph**
6. **Sentence fallback** (for oversized clauses exceeding the maximum limit)

**Provenance & Deterministic IDs:**
Every chunk is assigned a deterministic ID:
`{document_id}_p{page_start:03d}_c{sequence:03d}` (e.g. `rental_001_p004_c003`).

Chunk metadata schema:
```json
{
  "chunk_id": "trial_p001_c001",
  "document_id": "trial",
  "source_file": "trial.pdf",
  "page_start": 1,
  "page_end": 2,
  "section": "SECTION 1: PARTIES",
  "clause": "1.1",
  "text": "Landlord agrees to lease the premises...",
  "char_count": 450,
  "token_count": 85
}
```
*Section and clause are extracted directly from text and are `null` if not present—never invented.*

### 3. Local Embedding Strategy (`src/embedding/embedder.py`)
* **Model:** Default is `BAAI/bge-small-en-v1.5` (384-dimensional). It runs on CPU with high inference speed and top retrieval performance on retrieval benchmarks.
* **Offline guarantee:** The public model weights (~130 MB) are cached locally in `~/.cache/huggingface/hub` on first load. Afterwards, the entire pipeline operates with no network connection.
* **L2 Normalization:** Embeddings are L2-normalized so that inner product calculations represent exact Cosine Similarity.

### 4. FAISS Vector Store & Decoupled Metadata (`src/vector_store/faiss_store.py`)
To prevent storing multi-megabyte dense vector arrays redundantly in JSON files:
1. `index.faiss`: Native binary FAISS `IndexFlatIP` storing normalized dense vectors.
2. `mapping.json`: Maps integer vector ID (`"0"`, `"1"`, ...) to `chunk_id`.
3. `chunks.json`: Maps `chunk_id` to its full metadata and text.

---

## Running the Test Suite

Execute all 59 tests covering extraction, cleaning, chunking, embeddings, vector indexing, and pipeline integration:

```powershell
.venv311\Scripts\pytest.exe tests/ -v
```

Test coverage includes:
* Text extraction & page boundaries
* Scanned PDF detection & OCR fallback
* Legal negation & currency preservation
* Header/footer detection & line-wrap repair
* Chunk ID determinism, uniqueness, & sentence boundary integrity
* Embedding normalization & dimension verification
* FAISS vector addition, persistence, reload, & cosine retrieval accuracy
* End-to-end pipeline execution & CLI

---

## Privacy Guarantee

| Pipeline Stage | Network Calls | External APIs | Telemetry / Logging |
|---|---|---|---|
| PDF Extraction | None | None | Basename & page count only |
| Text Cleaning | None | None | Character ratio only |
| Semantic Chunking | None | None | Chunk count only |
| Embedding Inference | None (Local PyTorch) | None | Model name & dimension only |
| FAISS Indexing | None | None | Vector count only |

*Note: Model weights for the public embedding model are fetched once during initial setup and cached locally. No document text or user data is ever transmitted.*

---

## Future Integration with Gemma

In the next milestone, the retrieval layer will be connected to a local Gemma model:

```python
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore

# 1. User asks question
query = "What happens if rent is paid late?"

# 2. Embed question locally
embedder = LocalEmbedder()
query_vector = embedder.embed_texts([query])

# 3. Retrieve top-k chunks with provenance
store = FAISSVectorStore.load("data/vector_store")
top_chunks = store.search(query_vector, top_k=3)

# 4. Pass top_chunks to local Gemma for grounded explanation with exact page citation:
# "According to Section 4.1 on Page 2, a 5% penalty applies after 5 days."
```

---

## Local RAG query layer (new)

The first retrieval-and-generation layer is available in:

- `src/retrieval/retriever.py` — embeds a question, searches the supplied FAISS index, and returns relevant chunks or an uncertainty result.
- `src/generation/prompt_builder.py` — assembles a grounded prompt with page, section, clause, file, and chunk citations.
- `src/generation/gemma_client.py` — calls the OpenAI-compatible **local** LM Studio endpoint. It rejects non-loopback URLs as a privacy guard.
- `src/answer_question.py` — orchestrates retrieval, generation, and source metadata.
- `src/query_cli.py` — interactive CLI for querying one indexed document.

### Run it with LM Studio

1. Use CPython 3.11 and install dependencies: `python -m pip install -r requirements.txt`.
2. In LM Studio, download/load a compatible Gemma model and start its local server on `http://127.0.0.1:1234`.
3. In the project root, set the model ID shown by LM Studio if needed:
   - PowerShell: `$env:LM_STUDIO_MODEL="<model-id-from-LM-Studio>"`
4. Run: `python -m src.query_cli --store data/vector_store --top-k 5 --min-score 0.50`
5. Type a question and type `exit` to quit.

`--min-score 0.50` is only a starting setting, not a calibrated confidence probability. Evaluate it on representative questions before relying on it. The CLI expects the specified vector store to contain the one document being queried; do not combine unrelated contracts in one index without adding a document filter.

### CUAD compatibility

The existing parser accepts PDF input. CUAD JSON must not be passed to `pdf_extractor.py`. To evaluate with CUAD, add a dataset adapter that maps the selected dataset variant's contract text and annotations into evaluation records. Index contract text only when intentionally running a dataset-backed retrieval experiment. Keep the user's runtime contract isolated from CUAD and other contracts.

### Privacy notes

The new LM Studio client permits only HTTP loopback URLs (`localhost`, `127.0.0.1`, or `::1`) and uses Python's standard library. This is a guard against accidentally configuring a remote inference endpoint, not a full operating-system network sandbox. For strict offline verification, disconnect external networking or use OS-level firewall controls and confirm model weights are already downloaded. Avoid logging prompts or full contract text.

### Tests

Run `pytest -q`. The new RAG-layer tests use fake embeddings, retrieval results, and an LLM client, so they do not need to download a model or start LM Studio. The original tests still require the dependencies in `requirements.txt`.
