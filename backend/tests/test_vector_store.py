"""
test_vector_store.py
====================
pytest test suite for Module 5 — FAISS Vector Store.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.vector_store.faiss_store import FAISSVectorStore


class TestFAISSVectorStore:
    @pytest.fixture
    def sample_chunks_and_embeddings(self):
        chunks = [
            {
                "chunk_id": "doc_001_p001_c001",
                "document_id": "doc_001",
                "source_file": "doc_001.pdf",
                "page_start": 1,
                "page_end": 1,
                "section": "Rent",
                "clause": "1.1",
                "text": "The monthly rent is $1,000.",
                "char_count": 27,
                "token_count": 8,
            },
            {
                "chunk_id": "doc_001_p002_c002",
                "document_id": "doc_001",
                "source_file": "doc_001.pdf",
                "page_start": 2,
                "page_end": 2,
                "section": "Termination",
                "clause": "2.1",
                "text": "Termination requires 30 days notice.",
                "char_count": 36,
                "token_count": 9,
            },
            {
                "chunk_id": "doc_001_p003_c003",
                "document_id": "doc_001",
                "source_file": "doc_001.pdf",
                "page_start": 3,
                "page_end": 3,
                "section": "Pet Policy",
                "clause": "3.1",
                "text": "No pets allowed without landlord consent.",
                "char_count": 42,
                "token_count": 10,
            },
        ]
        # Create deterministic synthetic normalized embeddings (dim=4)
        raw = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return chunks, raw

    def test_add_documents_and_count(self, sample_chunks_and_embeddings):
        chunks, embeddings = sample_chunks_and_embeddings
        store = FAISSVectorStore(dimension=4)
        store.add_documents(chunks, embeddings)
        assert store.total_vectors == 3
        assert len(store.mapping) == 3
        assert len(store.chunks) == 3

    def test_vector_id_mapping(self, sample_chunks_and_embeddings):
        chunks, embeddings = sample_chunks_and_embeddings
        store = FAISSVectorStore(dimension=4)
        store.add_documents(chunks, embeddings)

        assert store.mapping[0] == "doc_001_p001_c001"
        assert store.mapping[1] == "doc_001_p002_c002"
        assert store.mapping[2] == "doc_001_p003_c003"

    def test_save_and_reload(self, sample_chunks_and_embeddings, tmp_path):
        chunks, embeddings = sample_chunks_and_embeddings
        store = FAISSVectorStore(dimension=4)
        store.add_documents(chunks, embeddings)

        idx_path, map_path, chk_path = store.save(tmp_path)
        assert idx_path.exists()
        assert map_path.exists()
        assert chk_path.exists()

        # Reload
        reloaded = FAISSVectorStore.load(tmp_path)
        assert reloaded.total_vectors == 3
        assert reloaded.dimension == 4
        assert reloaded.mapping[0] == "doc_001_p001_c001"
        assert reloaded.chunks["doc_001_p001_c001"]["text"] == "The monthly rent is $1,000."

    def test_search_accuracy(self, sample_chunks_and_embeddings):
        chunks, embeddings = sample_chunks_and_embeddings
        store = FAISSVectorStore(dimension=4)
        store.add_documents(chunks, embeddings)

        # Query aligned with vector 1 (Termination)
        query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        # Top result should be the termination chunk with similarity ~1.0
        assert results[0]["chunk_id"] == "doc_001_p002_c002"
        assert results[0]["section"] == "Termination"
        assert pytest.approx(results[0]["score"], abs=1e-5) == 1.0

    def test_dimension_mismatch_raises(self, sample_chunks_and_embeddings):
        chunks, _ = sample_chunks_and_embeddings
        store = FAISSVectorStore(dimension=4)
        bad_embeddings = np.ones((len(chunks), 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.add_documents(chunks, bad_embeddings)
