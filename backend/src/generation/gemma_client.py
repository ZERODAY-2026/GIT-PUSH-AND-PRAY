"""Minimal local-only client for LM Studio's OpenAI-compatible API."""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def _validate_local_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Privacy guard: LM Studio URL must use HTTP on localhost/127.0.0.1/::1")
    return base_url.rstrip("/")


class LocalGemmaClient:
    """Call a Gemma model already loaded in LM Studio or Ollama; never accepts remote URLs."""
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, temperature: float = 0.1,
                 max_tokens: int = 800) -> None:
        self.base_url = _validate_local_url(base_url or os.getenv("GEMMA_BASE_URL") or os.getenv("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL))
        self.model = model or os.getenv("GEMMA_MODEL") or os.getenv("LM_STUDIO_MODEL", "local-model")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def check_connection(self) -> dict[str, Any]:
        """Check if local inference server is reachable on localhost."""
        try:
            req = Request(self.base_url + "/models", headers={"Content-Type": "application/json"}, method="GET")
            with urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                return {"reachable": True, "models": models}
        except Exception:
            return {"reachable": False, "models": []}

    def _resolve_model_name(self) -> str:
        """Resolve model name, auto-detecting loaded model from server if 'local-model'."""
        if self.model and self.model != "local-model":
            return self.model
        conn = self.check_connection()
        if conn.get("reachable") and conn.get("models"):
            llm_models = [m for m in conn["models"] if "embed" not in m.lower()]
            if llm_models:
                self.model = llm_models[0]
                return self.model
        return self.model or "local-model"

    def generate(self, messages: list[dict[str, str]]) -> str:
        active_model = self._resolve_model_name()
        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(f"Local Gemma server returned HTTP {exc.code}: {err_body}") from exc
        except URLError as exc:
            raise RuntimeError("Could not reach local Gemma server at configured localhost URL. Start LM Studio / Ollama and load Gemma.") from exc

        try:
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            # Support reasoning models (Gemma 4 E2B/E4B, Gemma 3n) where thinking is in reasoning_content
            if not content.strip() and msg.get("reasoning_content"):
                reasoning = msg["reasoning_content"].strip()
                # If content is empty because reasoning consumed tokens, extract the conclusion
                content = reasoning
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected response format from local LLM chat completions endpoint") from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Local Gemma server returned an empty answer")

        answer = content.strip()
        disclaimer = "Not legal advice; preliminary document triage only."
        if disclaimer.lower() not in answer.lower():
            answer += "\n\n" + disclaimer
        return answer
