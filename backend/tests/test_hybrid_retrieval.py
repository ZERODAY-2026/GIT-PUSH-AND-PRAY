from src.retrieval.fusion import fuse_results
from src.retrieval.hybrid_retriever import HybridRetriever


def test_rrf_fuses_unique_chunk_ids_without_comparing_raw_scores():
    results = fuse_results(
        [
            {"chunk_id": "a", "dense_score": 0.2},
            {"chunk_id": "b", "dense_score": 0.9},
        ],
        [
            {"chunk_id": "b", "bm25_score": 100.0},
            {"chunk_id": "c", "bm25_score": 1.0},
        ],
        top_k=3,
        rrf_k=60,
    )
    assert [item["chunk_id"] for item in results] == ["b", "a", "c"]
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1


class FakeDense:
    store = type("Store", (), {"chunks": {"a": {"chunk_id": "a", "text": "termination"}}})()

    def retrieve(self, query, top_k):
        return [{"chunk_id": "a", "dense_score": 0.8}]


class FakeBM25:
    def retrieve(self, query, top_k):
        return [{"chunk_id": "a", "bm25_score": 2.0}]


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        return [{**candidates[0], "rerank_score": 0.9}]


def test_hybrid_retriever_returns_evidence_only():
    result = HybridRetriever(
        FakeDense(),
        FakeBM25(),
        FakeReranker(),
        initial_k=30,
        final_k=5,
    ).retrieve("What is termination?")
    assert result["status"] == "ok"
    assert result["results"][0]["chunk_id"] == "a"
    assert result["results"][0]["rerank_score"] == 0.9
    assert "answer" not in result
