"""
pipeline.py
===========
End-to-end local processing pipeline:
Extracted JSON -> Cleaning -> Chunking -> Local Embeddings -> Local FAISS Vector Store.

CLI usage:
---------
python -m src.processing.pipeline data/extracted/trial.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.embedding.embedder import DEFAULT_EMBEDDING_MODEL, LocalEmbedder
from src.processing.chunker import chunk_document, save_chunks
from src.processing.cleaner import clean_document, save_cleaned_document
from src.vector_store.faiss_store import FAISSVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("processing.pipeline")


def run_pipeline(
    extracted_json_path: str | Path,
    processed_dir: str | Path = "data/processed",
    vector_store_dir: str | Path = "data/vector_store",
    embedding_model_name: Optional[str] = None,
    target_tokens: int = 600,
    overlap_tokens: int = 90,
    remove_headers_footers: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full local pipeline on an extracted document JSON.

    Parameters
    ----------
    extracted_json_path : str | Path
        Path to the extracted JSON file from Module 1.
    processed_dir : str | Path
        Directory where cleaned and chunked JSON files are saved.
    vector_store_dir : str | Path
        Directory where the FAISS index and mapping files are stored.
    embedding_model_name : str, optional
        HuggingFace model name or local path (default: BAAI/bge-small-en-v1.5).
    target_tokens : int
        Target chunk size in tokens.
    overlap_tokens : int
        Overlap in tokens between adjacent chunks.
    remove_headers_footers : bool
        Whether to detect and strip repeated headers/footers.

    Returns
    -------
    dict
        Summary dictionary with processing metrics and saved file paths.
    """
    path = Path(extracted_json_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extracted JSON file not found: {path}")

    logger.info("Loading extracted document from: %s", path.name)
    with path.open("r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    source_file = extracted_data.get("source_file", path.name)
    total_pages = len(extracted_data.get("pages", []))
    total_chars_original = sum(
        len(p.get("text", "")) for p in extracted_data.get("pages", [])
    )

    if total_pages == 0 or total_chars_original == 0:
        raise ValueError(
            f"Extracted document '{source_file}' has no pages or text to process."
        )

    # -----------------------------------------------------------------------
    # Step 1: Cleaning
    # -----------------------------------------------------------------------
    logger.info("Step 1/4: Cleaning document text (conservative legal cleaning)...")
    cleaned_doc = clean_document(
        extracted_data, remove_headers_footers=remove_headers_footers
    )
    cleaned_path = save_cleaned_document(cleaned_doc, output_dir=processed_dir)

    # -----------------------------------------------------------------------
    # Step 2: Legal-aware Chunking
    # -----------------------------------------------------------------------
    logger.info(
        "Step 2/4: Chunking document (target=%d, overlap=%d)...",
        target_tokens,
        overlap_tokens,
    )
    chunks = chunk_document(
        cleaned_doc,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunks:
        raise ValueError(f"No chunks produced for document '{source_file}'.")

    chunks_path = save_chunks(chunks, source_file=source_file, output_dir=processed_dir)

    # Data integrity check
    total_chunk_chars = sum(c["char_count"] for c in chunks)
    logger.info(
        "Integrity check: Original: %d chars | Cleaned: %d chars | Chunks: %d chars",
        total_chars_original,
        cleaned_doc["total_chars_cleaned"],
        total_chunk_chars,
    )

    # -----------------------------------------------------------------------
    # Step 3: Local Embeddings
    # -----------------------------------------------------------------------
    logger.info("Step 3/4: Generating local embeddings on-device...")
    embedder = LocalEmbedder(model_name=embedding_model_name, normalize=True)
    embeddings = embedder.embed_chunks(chunks)

    # -----------------------------------------------------------------------
    # Step 4: Local FAISS Indexing & Persistence
    # -----------------------------------------------------------------------
    logger.info("Step 4/4: Building and persisting FAISS vector index...")
    store = FAISSVectorStore(dimension=embedder.dimension)
    store.add_documents(chunks, embeddings)
    index_path, mapping_path, meta_chunks_path = store.save(vector_store_dir)

    return {
        "source_file": source_file,
        "total_pages": total_pages,
        "cleaned_pages": len(cleaned_doc["pages"]),
        "chunks_created": len(chunks),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "vectors_indexed": store.total_vectors,
        "saved_files": {
            "cleaned_json": cleaned_path,
            "chunks_json": chunks_path,
            "faiss_index": index_path,
            "mapping_json": mapping_path,
            "vector_chunks_json": meta_chunks_path,
        },
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print clean summary to terminal without logging sensitive text."""
    print()
    print("=" * 60)
    print(f"Processing document: {summary['source_file']}")
    print("=" * 60)
    print(f"Pages:               {summary['total_pages']}")
    print(f"Cleaned pages:       {summary['cleaned_pages']}")
    print(f"Chunks created:      {summary['chunks_created']}")
    print(f"Embedding model:     {summary['embedding_model']}")
    print(f"Embedding dimension: {summary['embedding_dimension']}")
    print(f"Vectors indexed:     {summary['vectors_indexed']}")
    print()
    print("Saved:")
    for key, path in summary["saved_files"].items():
        print(f"  {path}")
    print("=" * 60)
    print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Local legal document processing pipeline (Cleaning -> Chunking -> Embeddings -> FAISS).",
    )
    parser.add_argument(
        "json_path",
        metavar="JSON_PATH",
        help="Path to the extracted JSON file, e.g. data/extracted/trial.json",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory to save cleaned and chunked JSON files (default: data/processed)",
    )
    parser.add_argument(
        "--vector-store-dir",
        default="data/vector_store",
        help="Directory to save FAISS index and mapping (default: data/vector_store)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=600,
        help="Target chunk size in tokens (default: 600)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=90,
        help="Overlap size in tokens (default: 90)",
    )
    parser.add_argument(
        "--keep-headers",
        action="store_true",
        help="Disable automatic repeated header/footer detection and removal",
    )

    args = parser.parse_args(argv)

    try:
        summary = run_pipeline(
            extracted_json_path=args.json_path,
            processed_dir=args.processed_dir,
            vector_store_dir=args.vector_store_dir,
            embedding_model_name=args.model,
            target_tokens=args.target_tokens,
            overlap_tokens=args.overlap_tokens,
            remove_headers_footers=not args.keep_headers,
        )
        _print_summary(summary)
        return 0
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
