"""External-service ports: embeddings and LLM completion.

Kept deliberately tiny (Interface Segregation). Fake implementations are used in
tests and offline runs; OpenAI-compatible implementations are the only place paid
inference happens, so they are easy to keep out of the hot loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """Turns text into vectors for semantic dedup/search (pgvector)."""

    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LlmProvider(ABC):
    """A minimal text-completion port used by the ingestion passes."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str: ...
