import json
from pathlib import Path
from src.processing.cleaner import clean_document
from src.processing.chunker import chunk_document
from src.embedding.embedder import LocalEmbedder
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import LegalRetriever

cuad_path = Path("data/cuad/test.json")
with open(cuad_path, "r", encoding="utf-8") as f:
    data = json.load(f)
c = data["data"][0]
paragraphs = c["paragraphs"][0]
qas = paragraphs["qas"]

embedder = LocalEmbedder()
pages = [{"page_number": 1, "text": paragraphs["context"]}]
chunks = chunk_document(clean_document({"source_file": "test", "total_pages": 1, "pages": pages}))
store = FAISSVectorStore(dimension=384)
store.add_documents(chunks, embedder.embed_chunks(chunks))
retriever = LegalRetriever(embedder, store, min_score=0.45)

print("Positive QA scores:")
for qa in qas:
    if not qa["is_impossible"] and qa["answers"]:
        cat = qa["question"].split('"')[1] if '"' in qa["question"] else qa["question"]
        res = retriever.retrieve(cat)
        print(f" POS: {cat[:25]:<25} score={res.best_score:.3f}")

print("\nNegative QA scores (sample of 10):")
neg_count = 0
for qa in qas:
    if qa["is_impossible"] or not qa["answers"]:
        cat = qa["question"].split('"')[1] if '"' in qa["question"] else qa["question"]
        res = retriever.retrieve(cat)
        print(f" NEG: {cat[:25]:<25} score={res.best_score:.3f}")
        neg_count += 1
        if neg_count >= 10:
            break
