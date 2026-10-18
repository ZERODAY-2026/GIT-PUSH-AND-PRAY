# Legal Triage

Privacy-first, local legal document analysis with a React frontend and a
Python/FastAPI backend. Documents, extracted text, embeddings, retrieval, and
model requests stay on the local machine. The system is for preliminary
document triage, not legal advice.

## Repository layout

```text
Legal/
├── backend/       Python API, ingestion, chunking, retrieval, and tests
├── frontend/      React + Vite user interface
├── evaluation/    Retrieval, LLM, end-to-end, and CUAD evaluation tools
├── .venv/         Shared local Python environment (ignored by Git)
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10–3.13
- Node.js 18 or newer
- Optional: LM Studio running a local Gemma model for answer generation

## Python environment

The project virtual environment is at the repository root, not inside
`backend`. From `Legal\backend`, activate it with:

```powershell
..\.venv\Scripts\Activate.ps1
```

If activation is blocked in the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
..\.venv\Scripts\Activate.ps1
```

You can also avoid activation:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start the backend

```powershell
cd C:\Users\HARI\Downloads\Legal\backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

The API is available at `http://127.0.0.1:8000`.

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `GET /api/health`
- Upload and index: `POST /api/documents/process`
- Existing question-answer flow: `POST /api/query`
- Evidence-only hybrid retrieval: `POST /api/retrieve`

The backend also includes a startup script:

```powershell
cd C:\Users\HARI\Downloads\Legal\backend
.\start_backend.bat
```

## Start the frontend

In a second terminal:

```powershell
cd C:\Users\HARI\Downloads\Legal\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend calls the backend at `http://127.0.0.1:8000`. Configure
additional frontend origins with `API_CORS_ORIGINS`, for example:

```powershell
$env:API_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"
```

## Local Gemma configuration

Start a model in LM Studio and its local OpenAI-compatible server on
`http://127.0.0.1:1234/v1`. Then set:

```powershell
$env:LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
$env:LM_STUDIO_MODEL = "google/gemma-4-e4b"
```

The backend rejects non-local LM Studio URLs.

## Processing pipeline

```text
PDF
  → PyMuPDF extraction
  → conservative cleaning
  → legal-aware section/clause chunking
  → local embeddings
  → FAISS index
  → retrieval
  → optional local Gemma answer
```

The existing retrieval-plus-answer path uses the configured embedding model
and FAISS index. The opt-in hybrid retrieval path adds BGE-M3, BM25, RRF, and
local cross-encoder reranking and returns evidence without generating an
answer.

Build the separate hybrid index from canonical chunks:

```powershell
cd C:\Users\HARI\Downloads\Legal\backend
..\.venv\Scripts\python.exe -m src.retrieval.build_index `
  --chunks data\vector_store\chunks.json `
  --output data\vector_store\retrieval
```

Query evidence only:

```powershell
..\.venv\Scripts\python.exe -m src.retrieval.query `
  "What is the termination notice period?"
```

## Tests

```powershell
cd C:\Users\HARI\Downloads\Legal\backend
..\.venv\Scripts\python.exe -m pytest tests -q
```

## Documentation

- [Architecture](./ARCHITECTURE.md)
- [Gemma and LM Studio guide](./GEMMA_GUIDE.md)
- [Privacy verification](./PRIVACY_VERIFICATION.md)
- [CUAD benchmark guide](./CUAD_BENCHMARK.md)
- [Evaluation README](./evaluation/README.md)

## Privacy and data handling

Local PDFs, extracted text, chunks, and vector indexes are ignored by Git.
Do not commit real legal documents, generated indexes, model weights, API
keys, or other sensitive data.

The system does not provide legal advice. Consult a qualified lawyer for
legal decisions.
