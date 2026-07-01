"""Backward-compatible aliases for the default Nous OpenAI-compatible adapter."""
from __future__ import annotations

from .providers_openai_compatible import OpenAICompatibleEmbedder, OpenAICompatibleLlm


class NousLlm(OpenAICompatibleLlm):
    """Backward-compatible alias for :class:`OpenAICompatibleLlm`."""


class NousEmbedder(OpenAICompatibleEmbedder):
    """Backward-compatible alias for :class:`OpenAICompatibleEmbedder`."""
