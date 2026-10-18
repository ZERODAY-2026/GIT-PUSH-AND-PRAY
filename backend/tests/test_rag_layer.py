import numpy as np
import pytest
from src.retrieval.retriever import LegalRetriever
from src.generation.prompt_builder import build_messages
from src.generation.gemma_client import LocalGemmaClient
from src.answer_question import answer_question

CHUNK = {
    "chunk_id": "contract_p004_c002", "source_file": "contract.pdf",
    "page_start": 4, "page_end": 4, "section": "TERMINATION", "clause": "7.2",
    "text": "Either party may terminate this agreement by giving 30 days written notice.",
    "score": 0.78,
}

class FakeEmbedder:
    def embed_texts(self, texts):
        self.texts = texts
        return np.ones((1, 384), dtype=np.float32)

class FakeStore:
    def __init__(self, results): self.results = results
    def search(self, vector, top_k=5): return self.results[:top_k]

class FakeLLM:
    def generate(self, messages):
        self.messages = messages
        return "The clause requires 30 days written notice. [SOURCE 1]\n\nNot legal advice; preliminary document triage only."

def test_retriever_keeps_relevant_chunks_and_uses_query_prefix():
    embedder = FakeEmbedder()
    retriever = LegalRetriever(embedder, FakeStore([CHUNK]), min_score=0.5)
    result = retriever.retrieve("Can the landlord terminate immediately?")
    assert result.status == "ok"
    assert result.chunks == [CHUNK]
    assert embedder.texts[0].startswith("Represent this sentence")

def test_retriever_rejects_low_score():
    chunk = {**CHUNK, "score": 0.2}
    result = LegalRetriever(FakeEmbedder(), FakeStore([chunk]), min_score=0.5).retrieve("termination notice")
    assert result.status == "low_relevance"
    assert result.chunks == []

def test_retriever_empty_question_rejected():
    with pytest.raises(ValueError):
        LegalRetriever(FakeEmbedder(), FakeStore([])).retrieve("  ")

def test_prompt_has_source_provenance_and_grounding_rules():
    messages = build_messages("What is the notice period?", [CHUNK])
    assert "Page 4" in messages[1]["content"]
    assert "Clause 7.2" in messages[1]["content"]
    assert "30 days written notice" in messages[1]["content"]
    assert "ONLY the supplied document excerpts" in messages[0]["content"]

def test_prompt_rejects_empty_evidence():
    with pytest.raises(ValueError): build_messages("Question?", [])

def test_prompt_budget_truncates_large_retrieved_evidence():
    large_chunk = {**CHUNK, "text": "contract obligation " * 5000}
    messages = build_messages("What is required?", [large_chunk] * 5)
    combined = "\n".join(message["content"] for message in messages)
    assert len(combined) <= 9000
    assert "[TRUNCATED]" in combined

def test_remote_llm_url_is_rejected():
    with pytest.raises(ValueError): LocalGemmaClient(base_url="https://example.com/v1")

def test_end_to_end_returns_sources():
    retriever = LegalRetriever(FakeEmbedder(), FakeStore([CHUNK]), min_score=0.5)
    llm = FakeLLM()
    result = answer_question("What notice is required?", retriever, llm)
    assert result["status"] == "ok"
    assert result["sources"][0]["clause"] == "7.2"
    assert "SOURCE 1" in llm.messages[1]["content"]

def test_end_to_end_does_not_call_llm_when_retrieval_is_weak():
    retriever = LegalRetriever(FakeEmbedder(), FakeStore([{**CHUNK, "score": 0.1}]), min_score=0.5)
    class FailLLM:
        def generate(self, messages): raise AssertionError("LLM should not be called")
    result = answer_question("What notice is required?", retriever, FailLLM())
    assert result["status"] == "low_relevance"
    assert result["sources"] == []
