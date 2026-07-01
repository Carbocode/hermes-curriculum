"""Deterministic, dependency-free fake providers.

These stand in for the OpenAI-compatible :class:`EmbeddingProvider` and
:class:`LlmProvider` adapters during tests and offline dry-runs. They exist so
that exercising the ingestion / dedup / selection paths costs nothing and never
touches the network: NO paid inference can happen behind a fake.

Two design rules make these safe to depend on:

* **Determinism.** Every output is a pure function of its inputs, derived via
  ``hashlib`` (and plain arithmetic), so the same text/prompt always yields the
  same vector/completion -- on any machine, in any process. There is no hidden
  state, no module-level randomness, and no wall-clock read.
* **Discrimination.** Distinct inputs produce distinct outputs with
  overwhelming probability (SHA-256 collision resistance), so a test that relies
  on "different concept -> different embedding" or "this prompt is not that
  prompt" holds.

Standard library only (``hashlib`` for the deterministic entropy, ``math`` for
the L2 norm), per the core-module constraint.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .ports.providers import EmbeddingProvider, LlmProvider

# Bytes consumed per embedding component. Four bytes -> a uint32, which is more
# than enough resolution for a fake vector and keeps the hash-stream math tidy.
_BYTES_PER_COMPONENT: int = 4
_UINT32_MAX: int = 0xFFFFFFFF


def _l2_normalise(vector: list[float]) -> list[float]:
    """Scale ``vector`` to unit Euclidean length.

    Normalising lets a cosine/dot-product nearest-neighbour search treat the
    fakes exactly like real embeddings. The zero-norm branch is a defensive
    guard: hash-derived components are essentially never all-zero, but dividing
    by a zero norm would raise, so we fall back to a fixed unit basis vector to
    keep the function total.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        fallback = [0.0] * len(vector)
        if fallback:
            fallback[0] = 1.0
        return fallback
    return [component / norm for component in vector]


class FakeEmbedder(EmbeddingProvider):
    """Hash-derived deterministic embedder.

    Each text is expanded into ``dim`` components by reading a SHA-256 keystream
    seeded with the text (counter-mode), mapping every 4-byte word into the
    signed range [-1, 1], then L2-normalising. Centering on zero spreads the
    components around the origin so that unrelated texts are roughly orthogonal,
    while identical texts collapse to the identical vector.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim < 1:
            # A zero/negative dimension cannot be normalised and would make
            # nearest-neighbour search meaningless; fail loudly at construction.
            raise ValueError("embedding dim must be >= 1")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit vector per input text, order preserved."""
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        """Deterministically map one text to a normalised ``dim``-vector."""
        return _l2_normalise(self._raw_components(text))

    def _raw_components(self, text: str) -> list[float]:
        """Expand ``text`` into ``dim`` signed floats via a SHA-256 keystream.

        We hash ``counter || text`` for an increasing counter and concatenate
        the digests until we have ``dim * 4`` bytes; this gives an arbitrarily
        long, fully deterministic byte stream from a single text. Prefixing the
        counter (rather than appending) keeps each block's input distinct even
        for the empty string.
        """
        text_bytes = text.encode("utf-8")
        needed = self.dim * _BYTES_PER_COMPONENT
        stream = bytearray()
        counter = 0
        while len(stream) < needed:
            block = hashlib.sha256(counter.to_bytes(4, "big") + text_bytes).digest()
            stream.extend(block)
            counter += 1
        components: list[float] = []
        for index in range(self.dim):
            start = index * _BYTES_PER_COMPONENT
            word = int.from_bytes(stream[start : start + _BYTES_PER_COMPONENT], "big")
            # Map [0, 2**32-1] -> [-1.0, 1.0] so components straddle zero.
            components.append((word / _UINT32_MAX) * 2.0 - 1.0)
        return components


class FakeLlm(LlmProvider):
    """Scripted, otherwise echo-stubbing deterministic completion provider.

    ``scripts`` maps a trigger substring to the canned completion to return when
    that substring appears anywhere in the prompt. This lets a test pin the
    output of a specific ingestion pass ("when the prompt mentions concept X,
    answer Y") without any model. When no trigger matches, a deterministic
    debug stub derived from the prompt is returned, so callers always get a
    stable, inspectable string.
    """

    def __init__(self, scripts: dict[str, str] | None = None) -> None:
        # Copy so a caller mutating their dict afterwards cannot change our
        # behaviour mid-run (determinism is a property of construction time).
        self.scripts: dict[str, str] = dict(scripts) if scripts else {}

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Return the scripted answer for ``prompt`` or a deterministic stub.

        Generation parameters (``system``/``max_tokens``/``temperature``) are
        accepted to satisfy the port but deliberately ignored: a fake must be
        deterministic, and in particular ``temperature`` introduces no
        randomness here.
        """
        scripted = self._match_script(prompt)
        if scripted is not None:
            return scripted
        return self._default_response(prompt)

    def _match_script(self, prompt: str) -> str | None:
        """Return the value of the most-specific trigger contained in ``prompt``.

        When several triggers match, the longest one wins (most specific),
        ties broken lexicographically. Choosing by content rather than by dict
        insertion order keeps the result independent of how ``scripts`` was
        built, so the match is fully deterministic.
        """
        matches = [key for key in self.scripts if key in prompt]
        if not matches:
            return None
        best = max(matches, key=lambda key: (len(key), key))
        return self.scripts[best]

    def _default_response(self, prompt: str) -> str:
        """Build a stable, human-readable stub keyed to the prompt.

        The SHA-256 prefix guarantees distinct prompts get distinct stubs (so a
        test can assert two prompts differ), while the whitespace-collapsed echo
        keeps the output debuggable when it shows up in a transcript.
        """
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        preview = " ".join(prompt.split())[:120]
        return f"[fake-llm] sha256={digest} echo={preview!r}"
