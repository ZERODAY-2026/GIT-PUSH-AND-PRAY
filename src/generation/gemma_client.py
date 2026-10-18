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
    """Call a model already loaded in LM Studio; never accepts remote URLs."""
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, temperature: float = 0.1,
                 max_tokens: int = 800) -> None:
        self.base_url = _validate_local_url(base_url or os.getenv("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL))
        self.model = model or os.getenv("LM_STUDIO_MODEL", "local-model")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
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
            raise RuntimeError(f"LM Studio returned HTTP {exc.code}; check that a model is loaded and the local server is running.") from exc
        except URLError as exc:
            raise RuntimeError("Could not reach LM Studio at the configured localhost URL. Start its local server and load Gemma.") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected response format from LM Studio chat completions endpoint") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LM Studio returned an empty answer")
        answer = content.strip()
        disclaimer = "Not legal advice; preliminary document triage only."
        if disclaimer.lower() not in answer.lower():
            answer += "\n\n" + disclaimer
        return answer
