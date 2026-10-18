"""Build the opt-in BGE-M3/BM25 retrieval index from existing chunks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore


def build_index(
    chunks_path: str | Path,
    output_dir: str | Path,
    *,
    embedding_model: str = "BAAI/bge-m3",
) -> dict[str, Any]:
    chunks_file = Path(chunks_path)
    output = Path(output_dir)
    payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks = list(payload.values()) if isinstance(payload, dict) else payload
    if not chunks:
        raise ValueError("No chunks found")

    embedder = LocalEmbedder(model_name=embedding_model, normalize=True)
    embeddings = embedder.embed_chunks(chunks)
    store = FAISSVectorStore(dimension=embedder.dimension)
    store.add_documents(chunks, embeddings)
    store.save(output)

    bm25_dir = output / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "corpus.json").write_text(
        json.dumps(
            {
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "texts": [chunk["text"] for chunk in chunks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metadata = {
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "normalize": True,
        "similarity_metric": "inner_product_cosine_on_normalized_vectors",
        "index_type": "faiss.IndexFlatIP",
        "created_at_unix": time.time(),
        "chunk_count": len(chunks),
    }
    (output / "index_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the opt-in hybrid retrieval index.")
    parser.add_argument("--chunks", default="data/vector_store/chunks.json")
    parser.add_argument("--output", default="data/vector_store/retrieval")
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()
    print(build_index(args.chunks, args.output, embedding_model=args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
