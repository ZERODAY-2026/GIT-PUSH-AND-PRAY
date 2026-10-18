# On-Device Privacy & Network Isolation Verification Guide

> **PS2 Criterion 4: On-Device Privacy and Network Isolation (15 pts — Safety Cap)**
> 
> *A step-by-step reproducible guide for judges to audit and confirm that zero private document text, chunks, embeddings, or questions leave the device.*

---

## 1. Quick Verification (Single Command)

Judges can verify complete network isolation and on-device execution by executing the automated audit script:

```bash
cd backend
..\.venv\Scripts\python.exe -m scripts.verify_offline_privacy
```

### What this script checks:
1. **Socket Interception:** Dynamically intercepts `socket.connect` during document ingestion, cleaning, chunking, embedding, indexing, retrieval, and answering. Asserts **0** outbound connections to non-localhost addresses.
2. **URL Guard Enforcement:** Attempts to pass cloud endpoints (OpenAI, Google Gemini Cloud, remote IPs) to `LocalGemmaClient` and verifies that **100%** are rejected with hard exceptions.
3. **Dependency Audit:** Scans all requirements to verify zero analytics, telemetry, or remote metric tracking libraries exist.
4. **Local Data Persistence:** Verifies all indices and extracted artifacts reside strictly within the project directory.

---

## 2. Air-Gapped / Offline Test (Network Disabled)

Judges can disable all network interfaces on the machine to confirm full offline autonomy:

### Step 1: Disconnect Internet
* Turn off Wi-Fi and unplug Ethernet (or in Linux/Docker: `--network none`).

### Step 2: Run End-to-End Processing
```bash
cd backend
..\.venv\Scripts\python.exe -m pytest tests/test_privacy_isolation.py -v
```

All tests will pass with **0 network access required**, because:
* PyMuPDF parsing runs 100% locally.
* Cleaners and chunkers run in-memory.
* Embedding model weights (`bge-small-en-v1.5`) are cached locally in the environment.
* FAISS runs on CPU/GPU via native C++ binaries.
* Gemma model runs locally in LM Studio or Ollama on `127.0.0.1`.

---

## 3. Passive Connection Monitoring / Wireshark Audit

Judges monitoring system sockets or using packet capture (Wireshark / tcpdump / Proxyman) will observe:

| Phase | Destination IP | Port | Payload Content | Status |
|---|---|---|---|---|
| Document Upload | `127.0.0.1` | `8000` | In-memory multipart PDF | **Local Only** |
| Text Extraction | N/A | N/A | None (in-process C library) | **No Network** |
| Embedding Generation | N/A | N/A | None (in-process PyTorch) | **No Network** |
| FAISS Index Search | N/A | N/A | None (in-process C++) | **No Network** |
| Gemma LLM Query | `127.0.0.1` | `1234` | Evidence Excerpts + Question | **Localhost Only** |
| Remote Web Endpoints | — | — | **Zero Bytes** | **Verified Absent** |

---

## 4. Code & Dependency Review for Telemetry

Run this automated test to verify absence of any tracking or analytics SDK:

```bash
cd backend
..\.venv\Scripts\python.exe -m pytest tests/test_privacy_isolation.py::test_dependency_audit_zero_telemetry
```

Verified absent: `wandb`, `sentry`, `segment`, `mixpanel`, `datadog`, `newrelic`, `snowflake`, `google-analytics`.

---

## 5. Temporary File and Data Storage Checks

All artifacts created during processing are saved locally:
* Uploaded PDFs: `backend/data/input/`
* Extracted page JSON: `backend/data/extracted/`
* Cleaned chunks: `backend/data/processed/`
* FAISS vector index & mappings: `backend/data/vector_store/`

No data is ever stored in world-readable temporary folders or uploaded to cloud storage buckets.
