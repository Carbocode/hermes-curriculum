"""The ingestion pipeline: an ordered Chain of Responsibility plus persistence.

A :class:`Pipeline` is just an ordered sequence of :class:`IngestionPass`
objects and the glue that runs them against a shared
:class:`IngestionContext`, then writes the verified results out through the
repository ports. Keeping "run the passes" and "persist the result" as two
separate steps lets a caller inspect (or dry-run) the working set before
committing anything to storage.

The composition is dependency-injected end to end: passes receive their
providers, ``persist`` receives its repositories. Nothing here knows about a
concrete adapter, so the whole pipeline is exercised in tests with the
deterministic ``FakeLlm``/``FakeEmbedder`` and the in-memory repositories. A
real run swaps in OpenAI-compatible providers and the Postgres/OKF repositories
-- the only place paid inference and durable writes happen -- without any
change to this module.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..ports.providers import EmbeddingProvider, LlmProvider
from ..ports.repositories import (
    ConceptIndexRepository,
    ContentRepository,
    EdgeRepository,
    QuestionRepository,
)
from .passes import (
    DedupePass,
    ExtractPass,
    InferEdgesPass,
    IngestionContext,
    IngestionPass,
    QuestionGenPass,
    VerifyPass,
)

__all__ = ["Pipeline", "default_pipeline"]


class Pipeline:
    """Runs a fixed, ordered chain of passes and persists the result.

    The pass order is the caller's responsibility (see :func:`default_pipeline`
    for the standard one); ``Pipeline`` only guarantees the passes execute
    left-to-right against the same context, which is what makes the chain a
    Chain of Responsibility rather than an unordered bag of transforms.
    """

    def __init__(self, passes: Sequence[IngestionPass]) -> None:
        # Snapshot into a tuple so a caller mutating their list afterwards
        # cannot reorder the chain mid-flight (determinism is fixed at build).
        self._passes: tuple[IngestionPass, ...] = tuple(passes)

    @property
    def passes(self) -> tuple[IngestionPass, ...]:
        """The ordered passes (read-only view, for introspection/tests)."""
        return self._passes

    def run(self, context: IngestionContext) -> IngestionContext:
        """Execute every pass in order against ``context`` and return it.

        The context is mutated in place; it is also returned so a caller can
        chain ``persist`` fluently on the same object.
        """
        for ingestion_pass in self._passes:
            ingestion_pass.run(context)
        return context

    def persist(
        self,
        context: IngestionContext,
        *,
        concepts: ConceptIndexRepository,
        edges: EdgeRepository,
        questions: QuestionRepository,
        content: ContentRepository,
    ) -> Mapping[str, int]:
        """Write the (already verified) working set out through the ports.

        Order matters: concept prose is written first so its content hash can be
        stamped onto the concept index row, linking structure (Postgres) to
        content (OKF) the way the sync layer expects. The same is done for
        questions. Derived embeddings, when present, are cached on the index.
        Returns the counts written, which is convenient for logging and tests.
        """
        for concept in context.concepts:
            stored = concept
            concept_content = context.concept_content_for(concept.id)
            if concept_content is not None:
                content_hash = content.put_concept_content(concept_content)
                # Point the index row at the just-written content (staleness
                # marker); Concept is frozen, so replace rather than mutate.
                stored = replace(stored, content_hash=content_hash)
            concepts.upsert(stored)
            vector = context.embeddings.get(concept.id)
            if vector is not None:
                concepts.set_embedding(concept.id, vector)

        for edge in context.edges:
            edges.upsert(edge)

        for question in context.questions:
            question_content = context.question_content_for(question.id)
            if question_content is not None:
                content.put_question_content(question_content)
            questions.upsert(question)

        return {
            "concepts": len(context.concepts),
            "edges": len(context.edges),
            "questions": len(context.questions),
        }


def default_pipeline(llm: LlmProvider, embedder: EmbeddingProvider) -> Pipeline:
    """Build the standard ingestion chain.

    Order is load-bearing: extract first, then dedup (so later passes never see
    duplicates), then infer edges over the deduped concepts, then generate
    questions over concepts and edges, and finally verify -- the grounding gate
    runs last so it can prune any concept/edge/question that survived the
    earlier passes without real provenance.

    ``llm``/``embedder`` are injected providers: deterministic fakes in tests,
    OpenAI-compatible adapters in a real run.
    """
    return Pipeline(
        [
            ExtractPass(llm),
            DedupePass(embedder),
            InferEdgesPass(llm),
            QuestionGenPass(llm),
            VerifyPass(),
        ]
    )
