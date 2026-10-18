"""HTTP API for the local legal document pipeline and question answering."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.answer_question import answer_question
from src.embedding.embedder import LocalEmbedder
from src.generation.gemma_client import LocalGemmaClient
from src.ingestion.pdf_extractor import extract_pdf, save_outputs
from src.processing.pipeline import run_pipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import LegalRetriever
from src.retrieval.reranker import LocalReranker
from src.vector_store.faiss_store import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

app = FastAPI(
    title="Local Legal RAG API",
    version="1.0.0",
    description="Local document indexing and grounded legal question answering.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.45, ge=-1.0, le=1.0)  # slightly lower default for broader recall
    model: str | None = None


class IndexRequest(BaseModel):
    extracted_json_path: str = Field(..., min_length=1)
    target_tokens: int = Field(default=600, ge=1, le=10000)
    overlap_tokens: int = Field(default=90, ge=0, le=5000)
    model: str | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    initial_k: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    model: str | None = None
    reranker_model: str | None = None
    debug: bool = False


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name.lower().endswith(".pdf") or not re.fullmatch(r"[\w .-]+", name):
        raise HTTPException(status_code=400, detail="Only PDF filenames are allowed.")
    return name


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Path must be inside the project directory.")
    return resolved


@lru_cache(maxsize=1)
def _load_query_components() -> tuple[FAISSVectorStore, LocalEmbedder]:
    store = FAISSVectorStore.load(VECTOR_STORE_DIR)
    embedder = LocalEmbedder()
    if embedder.dimension != store.dimension:
        raise RuntimeError(
            f"Embedding/index dimension mismatch: model={embedder.dimension}, index={store.dimension}"
        )
    return store, embedder


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return API and index availability without loading the embedding model."""
    index_exists = (VECTOR_STORE_DIR / "index.faiss").exists()
    return {
        "status": "ok",
        "index_ready": index_exists,
        "llm_base_url": os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Store a PDF locally for indexing; document contents are not returned."""
    filename = _safe_filename(file.filename or "")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = INPUT_DIR / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    destination.write_bytes(content)
    return {"status": "uploaded", "filename": filename, "path": str(destination.relative_to(PROJECT_ROOT))}


@app.post("/api/documents/process")
async def process_document(
    file: UploadFile = File(...),
    target_tokens: int = Query(default=600, ge=1, le=10000),
    overlap_tokens: int = Query(default=90, ge=0, le=5000),
    model: str | None = Query(default=None),
) -> dict[str, Any]:
    """Upload a PDF, extract it, and build its searchable local index."""
    filename = _safe_filename(file.filename or "")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = INPUT_DIR / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    pdf_path.write_bytes(content)

    try:
        extraction = extract_pdf(pdf_path)
        if extraction.error:
            raise ValueError(extraction.error)
        _, extracted_json_path = save_outputs(extraction, output_dir=EXTRACTED_DIR)
        summary = run_pipeline(
            extracted_json_path=extracted_json_path,
            processed_dir=PROCESSED_DIR,
            vector_store_dir=VECTOR_STORE_DIR,
            embedding_model_name=model,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        _load_query_components.cache_clear()
        return {
            "status": "indexed",
            "filename": filename,  # original PDF filename, used by frontend
            "summary": summary,    # contains total_pages, chunks_created, embedding_model, etc.
        }
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {type(exc).__name__}") from exc


@app.post("/api/documents/index")
def index_document(request: IndexRequest) -> dict[str, Any]:
    """Run extraction and indexing for a previously uploaded or local PDF."""
    extracted_path = _resolve_project_path(request.extracted_json_path)
    try:
        summary = run_pipeline(
            extracted_json_path=extracted_path,
            processed_dir=PROCESSED_DIR,
            vector_store_dir=VECTOR_STORE_DIR,
            embedding_model_name=request.model,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens,
        )
        _load_query_components.cache_clear()
        return {"status": "indexed", "summary": summary}
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {type(exc).__name__}") from exc


@app.post("/api/query")
def query_document(request: QueryRequest) -> dict[str, Any]:
    """Retrieve evidence and ask the local LM Studio model for an answer."""
    try:
        store, embedder = _load_query_components()
        retriever = LegalRetriever(
            embedder,
            store,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        llm = LocalGemmaClient(model=request.model)
        return answer_question(request.question, retriever, llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Vector index is not ready. Index a document first.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}") from exc


@app.post("/api/retrieve")
def retrieve_evidence(request: RetrievalRequest) -> dict[str, Any]:
    """Return hybrid-ranked evidence without calling Gemma or another LLM."""
    retrieval_dir = VECTOR_STORE_DIR / "retrieval"
    try:
        metadata_path = retrieval_dir / "index_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Hybrid retrieval index is not ready.")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedder = LocalEmbedder(model_name=request.model or metadata["embedding_model"])
        store = FAISSVectorStore.load(retrieval_dir)
        retriever = HybridRetriever(
            DenseRetriever(store, embedder),
            BM25Retriever(retrieval_dir / "bm25"),
            LocalReranker(request.reranker_model),
            initial_k=request.initial_k,
            final_k=request.top_k,
            rrf_k=request.rrf_k,
        )
        return retriever.retrieve(request.query, debug=request.debug)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Hybrid retrieval index is not ready. Build it with src.retrieval.build_index first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="Hybrid retrieval dependency is not installed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(exc).__name__}") from exc
