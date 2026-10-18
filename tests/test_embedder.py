"""
test_embedder.py
================
pytest test suite for Module 4 — Local Embedder.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.embedding.embedder import LocalEmbedder


class TestLocalEmbedder:
    def test_default_init(self):
        embedder = LocalEmbedder()
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.normalize is True

    def test_custom_init(self):
        embedder = LocalEmbedder(model_name="custom/model", normalize=False, device="cpu")
        assert embedder.model_name == "custom/model"
        assert embedder.normalize is False
        assert embedder.device == "cpu"

    def test_embed_texts_with_mock_model(self):
        embedder = LocalEmbedder(model_name="mock-model", normalize=True)

        mock_st = MagicMock()
        # Mock encode to return 3 vectors of dimension 4
        unnorm_vecs = np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        mock_st.encode.return_value = unnorm_vecs

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st):
            # Also mock test_vec during _load_model
            mock_st.encode.side_effect = [
                np.zeros((1, 4), dtype=np.float32),  # for test_vec in _load_model
                unnorm_vecs,  # for actual call
            ]
            vecs = embedder.embed_texts(["Text 1", "Text 2", "Text 3"])

            assert vecs.shape == (3, 4)
            # Check L2 normalization
            norms = np.linalg.norm(vecs, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3)

    def test_embed_empty_texts_returns_empty_array(self):
        embedder = LocalEmbedder(model_name="mock-model")
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((1, 4), dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st):
            vecs = embedder.embed_texts([])
            assert vecs.shape[0] == 0
            assert vecs.ndim == 2

    def test_embed_chunks_delegates_to_embed_texts(self):
        embedder = LocalEmbedder(model_name="mock-model")
        chunks = [
            {"chunk_id": "c1", "text": "Clause 1"},
            {"chunk_id": "c2", "text": "Clause 2"},
        ]
        with patch.object(embedder, "embed_texts") as mock_embed:
            mock_embed.return_value = np.zeros((2, 384), dtype=np.float32)
            res = embedder.embed_chunks(chunks)
            mock_embed.assert_called_once_with(["Clause 1", "Clause 2"], batch_size=32)
            assert res.shape == (2, 384)
