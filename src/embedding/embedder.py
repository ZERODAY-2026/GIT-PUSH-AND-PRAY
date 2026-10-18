"""
embedder.py
===========
Module 4 — Local sentence-transformers embeddings.

Responsibilities
----------------
- Generate dense vector embeddings for legal document chunks.
- Run completely locally on-device. No cloud APIs or network calls.
- Default model: BAAI/bge-small-en-v1.5 (dimension: 384, fast & high retrieval accuracy).
- Model name is configurable via parameter or EMBEDDING_MODEL environment variable.
- Normalize embeddings to unit length (L2) so inner product corresponds to cosine similarity.
- Support batching for efficiency.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger("embedding.embedder")

# Default embedding model: high performance, lightweight, 384-dimensional
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class LocalEmbedder:
    """
    On-device embedder using sentence-transformers.

    Attributes
    ----------
    model_name : str
        The Hugging Face model identifier or local path.
    dimension : int
        Vector dimensionality (e.g., 384 for bge-small-en-v1.5).
    normalize : bool
        Whether to L2-normalize vectors for cosine similarity via inner product.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.normalize = normalize
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy load sentence-transformers model on first use."""
        if self._model is not None:
            return self._model

        logger.info(
            "Loading local embedding model: '%s' (privacy guarantee: on-device execution)",
            self.model_name,
        )

        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._model = SentenceTransformer(self.model_name, **kwargs)
            # Determine dimensionality
            test_vec = self._model.encode(["test"], normalize_embeddings=self.normalize)
            self._dimension = test_vec.shape[1]
            logger.info(
                "Embedding model '%s' loaded successfully. Dimension: %d",
                self.model_name,
                self._dimension,
            )
            return self._model
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore[return-value]

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of text strings into a 2D numpy array of shape (N, D).

        Parameters
        ----------
        texts : list of str
            Chunk texts to encode.
        batch_size : int, optional
            Batch size for model inference (default: 32).
        show_progress : bool, optional
            Whether to show progress bar during inference.

        Returns
        -------
        np.ndarray
            Array of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            dim = self.dimension
            return np.empty((0, dim), dtype=np.float32)

        model = self._load_model()

        logger.info(
            "Encoding %d texts with '%s' (batch_size=%d, normalize=%s)...",
            len(texts),
            self.model_name,
            batch_size,
            self.normalize,
        )

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Sanity check
        if self.normalize and len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1)
            # Norms should be approximately 1.0
            if not np.allclose(norms, 1.0, atol=1e-3):
                logger.warning("Embeddings are not unit normalized. Normalizing now.")
                embeddings = embeddings / (norms[:, np.newaxis] + 1e-12)

        logger.info("Successfully generated embeddings shape: %s", embeddings.shape)
        return embeddings

    def embed_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract text from chunks and compute their embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)
