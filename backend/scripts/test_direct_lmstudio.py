import json
import urllib.request

payload = {
    "model": "google/gemma-4-e4b",
    "messages": [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Summarize in 1 sentence what an NDA is."}
    ],
    "max_tokens": 1000,
    "temperature": 0.1,
}
req = urllib.request.Request(
    "http://127.0.0.1:1234/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("LM Studio Success! Response:")
        print(json.dumps(res, indent=2))
except Exception as e:
    print("LM Studio Error:", e)
