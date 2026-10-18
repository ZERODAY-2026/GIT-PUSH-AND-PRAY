"""Inspect CUAD dataset structure and categories."""
import json
from pathlib import Path

cuad_path = Path(__file__).resolve().parent.parent / "data" / "cuad" / "test.json"
if not cuad_path.exists():
    cuad_path = Path(__file__).resolve().parent.parent / "data" / "cuad" / "CUADv1.json"

print(f"Loading {cuad_path.name}...")
with open(cuad_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Version:", data.get("version"))
contracts = data.get("data", [])
print(f"Total contracts in this split: {len(contracts)}")

if contracts:
    first = contracts[0]
    print(f"First contract title: {first.get('title')}")
    paragraphs = first.get("paragraphs", [])
    print(f"Number of paragraphs: {len(paragraphs)}")
    if paragraphs:
        p0 = paragraphs[0]
        context = p0.get("context", "")
        print(f"Context length: {len(context)} chars")
        qas = p0.get("qas", [])
        print(f"Number of questions/categories annotated: {len(qas)}")
        print("\nSample annotated questions/categories:")
        positive_qas = [q for q in qas if not q.get("is_impossible") and len(q.get("answers", [])) > 0]
        print(f"Categories with positive answers in this contract: {len(positive_qas)}")
        for q in positive_qas[:5]:
            ans = q["answers"][0]["text"]
            print(f" - Q: {q['question'][:80]}...")
            print(f"   A snippet: {ans[:100]}...\n")
