import json
import urllib.request

payload = {"question": "What are the termination conditions and notice periods?"}
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/query",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
print("Sending query to local backend + Gemma 4 E4B...")
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("\nQuery Status:", res.get("status"))
        print("Best Score:", res.get("best_score"))
        print("Sources Retrieved:", len(res.get("sources", [])))
        for s in res.get("sources", []):
            print(f" - Page {s.get('page_start')} | Clause {s.get('clause')} | Score: {s.get('score'):.3f}")
        print("\n" + "=" * 60)
        print("GEMMA 4 E4B ANSWER:")
        print("=" * 60)
        print(res.get("answer"))
        print("=" * 60)
except Exception as e:
    print("Error querying backend:", e)
