"""
test_pipeline.py
================
pytest test suite for Module 2–5 Local Processing Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.processing.pipeline import run_pipeline, main


class TestPipelineEndToEnd:
    @pytest.fixture
    def sample_extracted_json(self, tmp_path):
        data = {
            "source_file": "test_contract.pdf",
            "source_path": str(tmp_path / "test_contract.pdf"),
            "total_pages": 2,
            "total_chars": 800,
            "scanned_pages": 0,
            "extraction_error": None,
            "pages": [
                {
                    "page_number": 1,
                    "char_count": 400,
                    "needs_ocr": False,
                    "text": (
                        "CONFIDENTIAL CONTRACT\nPage 1 of 2\n\n"
                        "SECTION 1: SERVICES AND TERM\n\n"
                        "1.1 Contractor shall provide software consulting services.\n"
                        "1.2 Term shall be 12 months from signing date.\n\n"
                        "Footer boilerplate"
                    ),
                },
                {
                    "page_number": 2,
                    "char_count": 400,
                    "needs_ocr": False,
                    "text": (
                        "CONFIDENTIAL CONTRACT\nPage 2 of 2\n\n"
                        "SECTION 2: COMPENSATION\n\n"
                        "2.1 Client shall pay $10,000 upon delivery of milestone 1.\n"
                        "2.2 Late invoices shall accrue interest at 1.5% per month.\n\n"
                        "Footer boilerplate"
                    ),
                },
            ],
        }
        json_path = tmp_path / "test_contract.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        return json_path

    def test_run_pipeline_creates_all_artifacts(self, sample_extracted_json, tmp_path):
        processed_dir = tmp_path / "processed"
        vector_store_dir = tmp_path / "vector_store"

        # Mock embedder to produce fast deterministic vectors without downloading model
        with patch("src.processing.pipeline.LocalEmbedder") as MockEmbedder:
            instance = MockEmbedder.return_value
            instance.dimension = 4
            instance.model_name = "mock/bge-small"

            def mock_embed(chunks):
                n = len(chunks)
                arr = np.random.randn(n, 4).astype(np.float32)
                arr /= np.linalg.norm(arr, axis=1, keepdims=True)
                return arr

            instance.embed_chunks.side_effect = mock_embed

            summary = run_pipeline(
                extracted_json_path=sample_extracted_json,
                processed_dir=processed_dir,
                vector_store_dir=vector_store_dir,
                target_tokens=100,
            )

        assert summary["total_pages"] == 2
        assert summary["chunks_created"] > 0
        assert summary["vectors_indexed"] == summary["chunks_created"]

        # Check saved files exist
        assert summary["saved_files"]["cleaned_json"].exists()
        assert summary["saved_files"]["chunks_json"].exists()
        assert summary["saved_files"]["faiss_index"].exists()
        assert summary["saved_files"]["mapping_json"].exists()
        assert summary["saved_files"]["vector_chunks_json"].exists()

        # Check mapping and chunks consistency
        with summary["saved_files"]["mapping_json"].open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        with summary["saved_files"]["vector_chunks_json"].open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        assert len(mapping) == summary["chunks_created"]
        assert len(chunks) == summary["chunks_created"]
        # Check vector 0 maps to a valid chunk ID
        assert "0" in mapping
        first_cid = mapping["0"]
        assert first_cid in chunks
        assert "text" in chunks[first_cid]
        assert "page_start" in chunks[first_cid]

    def test_pipeline_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pipeline(tmp_path / "nonexistent.json")

    def test_pipeline_empty_doc_raises(self, tmp_path):
        bad_json = tmp_path / "empty.json"
        bad_json.write_text(json.dumps({"source_file": "empty.pdf", "pages": []}))
        with pytest.raises(ValueError, match="no pages or text"):
            run_pipeline(bad_json)

    def test_cli_invocation(self, sample_extracted_json, tmp_path):
        processed_dir = tmp_path / "processed"
        vector_store_dir = tmp_path / "vector_store"

        with patch("src.processing.pipeline.LocalEmbedder") as MockEmbedder:
            instance = MockEmbedder.return_value
            instance.dimension = 4
            instance.model_name = "mock/bge-small"
            instance.embed_chunks.side_effect = lambda chunks: np.ones((len(chunks), 4), dtype=np.float32)

            exit_code = main(
                [
                    str(sample_extracted_json),
                    "--processed-dir",
                    str(processed_dir),
                    "--vector-store-dir",
                    str(vector_store_dir),
                ]
            )
            assert exit_code == 0
            assert (processed_dir / "test_contract_cleaned.json").exists()
