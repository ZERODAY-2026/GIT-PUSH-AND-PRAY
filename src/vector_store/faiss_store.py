"""
faiss_store.py
==============
Module 5 — Local FAISS Vector Store and Metadata Mapping.

Responsibilities
----------------
- Manage on-device vector index using FAISS (IndexFlatIP for cosine similarity).
- Persist:
  1. index.faiss: Fast binary vector index
  2. mapping.json: Integer vector ID -> chunk_id mapping
  3. chunks.json: chunk_id -> metadata dictionary (provenance, section, text)
- Keep vector embeddings in FAISS and document metadata in JSON (no duplicate embeddings).
- Support deterministic rebuilds and cross-session persistence.
- Provide clean search() API designed for future Gemma RAG querying.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vector_store.faiss_store")

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None  # Handled with friendly error message


class FAISSVectorStore:
    """
    On-device FAISS vector index with decoupled metadata mapping.

    Parameters
    ----------
    dimension : int
        Dimensionality of the vector space (e.g. 384 for bge-small-en-v1.5).
    """

    INDEX_FILENAME = "index.faiss"
    MAPPING_FILENAME = "mapping.json"
    CHUNKS_FILENAME = "chunks.json"

    def __init__(self, dimension: int):
        if faiss is None:
            raise ImportError(
                "faiss is required for the local vector store. "
                "Install it with: pip install faiss-cpu"
            )
        self.dimension = dimension
        # IndexFlatIP calculates exact inner product (equivalent to cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        # Vector ID (0, 1, 2...) -> chunk_id
        self.mapping: Dict[int, str] = {}
        # chunk_id -> full chunk metadata
        self.chunks: Dict[str, Dict[str, Any]] = {}

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add chunks and their pre-computed embeddings to the index.

        Parameters
        ----------
        chunks : list of dict
            Chunk dictionaries containing chunk_id, text, metadata.
        embeddings : np.ndarray
            2D numpy array of shape (len(chunks), dimension), dtype float32.
        """
        if len(chunks) == 0:
            logger.warning("No chunks provided to add to vector store.")
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but embeddings has {embeddings.shape[0]} rows"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        start_idx = self.index.ntotal

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Update mapping and metadata
        for i, chunk in enumerate(chunks):
            vector_id = start_idx + i
            cid = chunk["chunk_id"]
            self.mapping[vector_id] = cid
            self.chunks[cid] = chunk

        logger.info(
            "Added %d documents to FAISS index. Total vectors in index: %d",
            len(chunks),
            self.index.ntotal,
        )

    def save(self, directory: str | Path) -> Tuple[Path, Path, Path]:
        """
        Persist the FAISS index, mapping, and chunk metadata to a local directory.

        Returns
        -------
        Tuple of (index_path, mapping_path, chunks_path)
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / self.INDEX_FILENAME
        mapping_path = out_dir / self.MAPPING_FILENAME
        chunks_path = out_dir / self.CHUNKS_FILENAME

        # Write FAISS binary index
        faiss.write_index(self.index, str(index_path))

        # Write vector ID -> chunk ID mapping (keys serialized as strings for JSON)
        str_mapping = {str(k): v for k, v in self.mapping.items()}
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(str_mapping, f, ensure_ascii=False, indent=2)

        # Write chunk ID -> chunk metadata (no duplicate embeddings!)
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        logger.info(
            "Vector store persisted to: %s (%d vectors)",
            out_dir,
            self.total_vectors,
        )

        return index_path, mapping_path, chunks_path

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load a persisted FAISSVectorStore from disk."""
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")

        in_dir = Path(directory)
        index_path = in_dir / cls.INDEX_FILENAME
        mapping_path = in_dir / cls.MAPPING_FILENAME
        chunks_path = in_dir / cls.CHUNKS_FILENAME

        for p in (index_path, mapping_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required vector store file: {p}")

        index = faiss.read_index(str(index_path))
        dimension = index.d

        store = cls(dimension=dimension)
        store.index = index

        with mapping_path.open("r", encoding="utf-8") as f:
            str_mapping = json.load(f)
            store.mapping = {int(k): v for k, v in str_mapping.items()}

        with chunks_path.open("r", encoding="utf-8") as f:
            store.chunks = json.load(f)

        logger.info(
            "Vector store loaded from %s (%d vectors, dim=%d)",
            in_dir,
            store.total_vectors,
            dimension,
        )
        return store

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query the FAISS index and return top-k matching chunks with similarity scores.
        Designed for seamless future consumption by the Gemma RAG layer.

        Parameters
        ----------
        query_vector : np.ndarray
            1D array of shape (dimension,) or 2D array of shape (1, dimension).
        top_k : int, optional
            Number of nearest chunks to retrieve (default: 5).

        Returns
        -------
        list of dict
            List of retrieved chunks with 'score' and complete provenance fields.
        """
        if self.total_vectors == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec[np.newaxis, :]

        if q_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {q_vec.shape[1]} does not match index dimension {self.dimension}"
            )

        k = min(top_k, self.total_vectors)
        distances, indices = self.index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.mapping.get(int(idx))
            if not chunk_id:
                continue
            chunk_data = self.chunks.get(chunk_id, {})
            # Return full metadata plus search score
            result_item = dict(chunk_data)
            result_item["score"] = float(dist)
            results.append(result_item)

        return results
