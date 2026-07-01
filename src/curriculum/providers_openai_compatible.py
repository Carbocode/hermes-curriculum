"""OpenAI-compatible provider adapters.

These adapters target providers that implement the common OpenAI HTTP shapes:
``POST /chat/completions`` and ``POST /embeddings``. Nous remains the default
base URL in settings for backwards compatibility, but any compatible vendor can
be configured by changing the base URL, API key, and model names.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Sequence

from .ports.providers import EmbeddingProvider, LlmProvider


class OpenAICompatibleLlm(LlmProvider):
    """Chat-completions-backed LLM provider for OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://inference-api.nousresearch.com/v1",
        model: str = "deepseek/deepseek-v4-flash",
        timeout: float = 300.0,
    ) -> None:
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout
        self.calls = 0
        self.failures = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            self.failures += 1
            detail = exc.read().decode("utf-8", "ignore")[:200]
            print(f"[OpenAICompatibleLlm] HTTP {exc.code}: {detail}", file=sys.stderr)
            return ""
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover
            self.failures += 1
            print(f"[OpenAICompatibleLlm] request failed: {exc}", file=sys.stderr)
            return ""
        try:
            return payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):  # pragma: no cover
            self.failures += 1
            print(
                f"[OpenAICompatibleLlm] unexpected response shape: {str(payload)[:200]}",
                file=sys.stderr,
            )
            return ""


class OpenAICompatibleEmbedder(EmbeddingProvider):
    """Embeddings provider for OpenAI-compatible ``/embeddings`` APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://inference-api.nousresearch.com/v1",
        model: str = "google/gemini-embedding-2",
        dim: int = 3072,
        timeout: float = 120.0,
        batch: int = 64,
    ) -> None:
        self._key = api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self.dim = dim
        self._timeout = timeout
        self._batch = max(1, batch)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Empty text confuses some embedding models; substitute a single space.
        items = [t if t and t.strip() else " " for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(items), self._batch):
            out.extend(self._embed_batch(items[i : i + self._batch]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
        return [list(row["embedding"]) for row in rows]
