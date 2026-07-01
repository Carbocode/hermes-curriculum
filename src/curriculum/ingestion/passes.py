"""Multipass ingestion: the working context plus the individual passes.

The ingestion pipeline is a Chain of Responsibility (see ``pipeline.py``): a
fixed, ordered sequence of small passes, each of which reads and mutates a
single shared :class:`IngestionContext`. Splitting extraction, dedup, edge
inference, question generation and verification into separate passes keeps each
one to a single responsibility and lets the pipeline be reordered or extended
(Open/Closed) without touching the others.

Determinism and cost
---------------------
Every pass takes its collaborators (an :class:`LlmProvider`, an
:class:`EmbeddingProvider`) by injection, so the same context plus the same
fakes always yields the same result -- no module-level randomness, no wall
clock, no network. In tests the collaborators are the deterministic
:class:`curriculum.providers_fake.FakeLlm`/``FakeEmbedder``; in a real run they
are the OpenAI-compatible adapters (the only place paid inference happens).

Grounding (anti-fabrication)
----------------------------
The whole graph is built from source chunks that carry a ``file``/``line``
provenance. The passes pass that provenance through faithfully -- in
particular :class:`ExtractPass` never invents a citation; if the model fails to
cite a source the candidate is left ungrounded on purpose, so the adversarial
:class:`VerifyPass` can drop it. Nothing reaches persistence that cannot be
traced to a source file present in the ingest input.

Standard library only (``json``/``re``/``math``/``dataclasses``), per the
core-module constraint.
"""
from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..domain.entities import (
    Concept,
    ConceptContent,
    Edge,
    Question,
    QuestionContent,
    SourceRef,
)
from ..domain.enums import EdgeType
from ..ports.providers import EmbeddingProvider, LlmProvider

__all__ = [
    "IngestionContext",
    "IngestionPass",
    "ExtractPass",
    "DedupePass",
    "InferEdgesPass",
    "QuestionGenPass",
    "VerifyPass",
]

# Tag stamped on every generated Question so its provenance (which pass minted
# it) is auditable downstream. Kept ASCII and stable -- it is persisted.
_GENERATED_BY = "ingestion.QuestionGenPass"


# --------------------------------------------------------------------------- #
# Small pure helpers (parsing the LLM JSON contract + vector math).
# --------------------------------------------------------------------------- #
def _clamp(value: float, lo: float, hi: float) -> float:
    """Bound ``value`` into ``[lo, hi]`` so domain invariants (importance and
    weight live in 0..1, difficulty in 1..5) hold even when the model returns a
    nonsense magnitude."""
    return max(lo, min(hi, value))


def _as_float(value: Any, default: float) -> float:
    """Best-effort float coercion. A malformed field must not abort the whole
    ingest, so we fall back to a sane default rather than raise."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    """Best-effort int coercion (same defensive contract as :func:`_as_float`)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(text: str) -> str:
    """Derive a stable, ASCII, path-like id from free text.

    Used only as a fallback when the model omits a concept id: lowercase, runs
    of non-alphanumeric characters collapse to a single hyphen, ends trimmed.
    Deterministic so the same title always yields the same id.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _to_source_ref(value: Any) -> SourceRef | None:
    """Build one :class:`SourceRef` from a ``{"file": ..., "line": ...}`` dict.

    Returns ``None`` when the dict is malformed or carries no ``file`` -- a ref
    without a file cannot ground anything, so it is dropped rather than kept as
    a half-citation.
    """
    if not isinstance(value, dict):
        return None
    file = value.get("file")
    if not isinstance(file, str) or not file:
        return None
    line = value.get("line")
    return SourceRef(file=file, line=line if isinstance(line, int) else None)


def _to_source_refs(value: Any) -> tuple[SourceRef, ...]:
    """Build a tuple of :class:`SourceRef` from a list of ref dicts.

    Non-list inputs and malformed entries are silently skipped; the result may
    legitimately be empty (an ungrounded candidate), which is what lets
    :class:`VerifyPass` exercise its grounding gate.
    """
    if not isinstance(value, list):
        return ()
    refs = [_to_source_ref(item) for item in value]
    return tuple(ref for ref in refs if ref is not None)


def _union_refs(
    primary: Sequence[SourceRef], extra: Sequence[SourceRef]
) -> tuple[SourceRef, ...]:
    """Merge two ref sequences preserving order and dropping duplicates.

    Used when DedupePass folds a near-duplicate into its representative: the
    surviving concept must inherit every distinct source the duplicate cited so
    no provenance is lost in the merge. Order is stable (primary first) so the
    result is deterministic.
    """
    out: list[SourceRef] = []
    seen: set[tuple[str, int | None]] = set()
    for ref in list(primary) + list(extra):
        key = (ref.file, ref.line)
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return tuple(out)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either vector has zero magnitude.

    ``zip`` truncates to the shorter length so a stray dimension mismatch is
    total rather than fatal. Defined locally (not imported from storage) so the
    ingestion module owns its own dedup maths.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _loads_lenient(text: str) -> Any:
    """Parse JSON that may be wrapped in prose, returning ``None`` on failure.

    Real models often bracket their JSON with explanatory text or markdown
    fences; the fakes emit bare JSON. We try a strict parse first, then fall
    back to slicing out the outermost ``{...}`` / ``[...]`` span. A total parse
    failure yields ``None`` so the caller treats that completion as "no items"
    instead of crashing the run.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _items(text: str, key: str) -> list[dict[str, Any]]:
    """Extract a list of item dicts from an LLM completion.

    The JSON contract accepts either a bare array or an object that wraps the
    array under ``key`` (e.g. ``{"concepts": [...]}``), so a slightly different
    model formatting still parses. Anything else (including unparseable output)
    yields an empty list -- best-effort ingestion, never an exception.
    """
    obj = _loads_lenient(text)
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]
    return []


# --------------------------------------------------------------------------- #
# The shared working context (the "WorkingSet").
# --------------------------------------------------------------------------- #
@dataclass
class IngestionContext:
    """Mutable accumulator threaded through every pass.

    NOT frozen: a Chain of Responsibility mutates one shared object as it flows
    down the chain. ``course`` is required because a :class:`Concept` is
    course-scoped and the structure must know which course it belongs to.
    Content lives alongside structure here (``concept_content``/
    ``question_content``) so a pass can ground a concept by its prose; the
    repositories split them apart again at persist time (the OKF/Postgres
    split).
    """

    course: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    concept_content: list[ConceptContent] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    question_content: list[QuestionContent] = field(default_factory=list)
    # Derived content embeddings keyed by surviving concept id, populated by
    # DedupePass and persisted as the pgvector cache. A derived cache, not part
    # of the OKF source of truth, hence it rides on the context, not in content.
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def concept_content_for(self, concept_id: str) -> ConceptContent | None:
        """Return the accumulated prose for ``concept_id`` (linear scan; the
        working set is small)."""
        for content in self.concept_content:
            if content.concept_id == concept_id:
                return content
        return None

    def question_content_for(self, question_id: str) -> QuestionContent | None:
        """Return the accumulated prose for ``question_id``."""
        for content in self.question_content:
            if content.question_id == question_id:
                return content
        return None

    def source_files(self) -> set[str]:
        """The set of legitimate provenance files (one per chunk).

        A citation is only trustworthy if it points at a file that was actually
        fed to this ingest run; VerifyPass uses this set to reject hallucinated
        references.
        """
        return {
            chunk["file"]
            for chunk in self.chunks
            if isinstance(chunk.get("file"), str) and chunk["file"]
        }


# --------------------------------------------------------------------------- #
# The internal pass ABC.
# --------------------------------------------------------------------------- #
class IngestionPass(ABC):
    """One link in the ingestion chain.

    Internal to the ingestion package on purpose: passes are an implementation
    detail of the pipeline, not a public port the application depends on. Each
    pass mutates the shared context in place and returns nothing.
    """

    name: str = "abstract"

    @abstractmethod
    def run(self, ctx: IngestionContext) -> None:
        """Read from and mutate ``ctx``. Must be deterministic."""


# --------------------------------------------------------------------------- #
# Pass 1: extract candidate concepts from each source chunk.
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = (
    "You extract atomic concepts from course material. Extract ONLY substantive, "
    "exam-relevant concepts -- not every sentence, command, or procedural step; "
    "prefer a few well-formed concepts over many trivial ones. Cite every concept "
    "with source_refs (file and line) taken ONLY from the provided chunk; never "
    "invent a citation. Reply with JSON only."
)


class ExtractPass(IngestionPass):
    """Prompt the LLM per chunk and turn the JSON answer into concepts.

    One LLM call per chunk keeps each prompt small and the provenance obvious
    (the chunk's own file/line). The JSON contract is::

        {"concepts": [{"id": "okf/path", "title": "...", "description": "...",
                       "body": "...", "importance": 0.0-1.0,
                       "source_refs": [{"file": "...", "line": int}]}]}

    We deliberately do NOT backfill the chunk's own ref when the model omits
    citations: laundering an uncited concept as grounded would defeat the
    anti-fabrication guarantee. Uncited concepts flow on ungrounded and are
    dropped later by VerifyPass.
    """

    name = "extract"

    def __init__(self, llm: LlmProvider) -> None:
        self._llm = llm

    def run(self, ctx: IngestionContext) -> None:
        for chunk in ctx.chunks:
            raw = self._llm.complete(
                self._prompt(chunk), system=_EXTRACT_SYSTEM, temperature=0.0
            )
            for item in _items(raw, "concepts"):
                built = self._build(item, ctx.course)
                if built is None:
                    continue
                concept, content = built
                ctx.concepts.append(concept)
                ctx.concept_content.append(content)

    def _prompt(self, chunk: dict[str, Any]) -> str:
        """Build the per-chunk extraction prompt (includes the chunk text and
        its provenance so the model can cite accurately)."""
        return (
            "Extract the atomic concepts taught in the following source chunk.\n"
            f"Source file: {chunk.get('file')} line: {chunk.get('line')}\n"
            "---\n"
            f"{chunk.get('text', '')}\n"
            "---\n"
            'Return JSON: {"concepts": [{"id": "short-kebab-slug", "title": "...", '
            '"description": "...", "body": "...", "importance": 0.0-1.0, '
            '"source_refs": [{"file": "...", "line": int}]}]}. '
            "The id is a SHORT kebab-case slug of the concept itself (no file "
            "names, no line numbers). Every concept MUST include at least one "
            "source_ref from this chunk."
        )

    def _build(
        self, item: dict[str, Any], course: str
    ) -> tuple[Concept, ConceptContent] | None:
        """Build a (Concept, ConceptContent) pair from one JSON object.

        Returns ``None`` when there is no usable identity (neither id nor
        title): such an item cannot be referenced or grounded, so it is dropped.
        """
        title = str(item.get("title") or "").strip()
        # Normalize to a consistent ``<course>/<slug>`` id. The model is
        # inconsistent about id namespaces (it sometimes echoes the chunk
        # file/line into the id), so derive the id deterministically from the
        # last segment of its proposed id (or the title) under one course prefix.
        raw_id = str(item.get("id") or "").strip()
        base = raw_id.rsplit("/", 1)[-1] if raw_id else title
        slug = _slug(base) or _slug(title)
        if not slug:
            return None
        course_slug = _slug(course)
        concept_id = f"{course_slug}/{slug}" if course_slug else slug
        description = str(item.get("description") or "").strip()
        body = str(item.get("body") or "").strip() or description or title
        importance = _clamp(_as_float(item.get("importance"), 0.5), 0.0, 1.0)
        refs = _to_source_refs(item.get("source_refs"))
        display_title = title or concept_id
        concept = Concept(
            id=concept_id,
            course=course,
            title=display_title,
            description=description,
            importance=importance,
            source_refs=refs,
        )
        content = ConceptContent(
            concept_id=concept_id,
            title=display_title,
            body=body,
            description=description,
            source_refs=refs,
        )
        return concept, content


# --------------------------------------------------------------------------- #
# Pass 2: merge semantically duplicate concepts.
# --------------------------------------------------------------------------- #
class DedupePass(IngestionPass):
    """Collapse concepts whose embeddings are near-identical.

    Extraction over overlapping chunks naturally produces the same concept more
    than once; persisting both would split a learner's mastery across phantom
    duplicates. We embed each concept's body, then greedily fold any concept
    whose cosine similarity to an already-kept representative is at or above
    ``threshold`` into that representative, unioning their source_refs so no
    provenance is lost. Runs before edge inference and question generation so
    those passes never see (or wire up) a duplicate. The surviving
    representative's embedding is stashed on the context for persistence as the
    pgvector cache.
    """

    name = "dedupe"

    def __init__(self, embedder: EmbeddingProvider, *, threshold: float = 0.92) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def run(self, ctx: IngestionContext) -> None:
        if not ctx.concepts:
            return
        vectors = self._embedder.embed([self._text_for(ctx, c) for c in ctx.concepts])

        kept: list[Concept] = []  # surviving representatives, in first-seen order
        kept_vectors: list[list[float]] = []
        # concept_id -> the representative it merged into (identity if kept).
        survivor_of: dict[str, str] = {}

        for concept, vector in zip(ctx.concepts, vectors):
            match_index = self._first_match(vector, kept_vectors)
            if match_index is None:
                kept.append(concept)
                kept_vectors.append(vector)
                survivor_of[concept.id] = concept.id
                ctx.embeddings[concept.id] = vector
            else:
                rep = kept[match_index]
                kept[match_index] = replace(
                    rep, source_refs=_union_refs(rep.source_refs, concept.source_refs)
                )
                survivor_of[concept.id] = rep.id

        ctx.concepts = kept
        ctx.concept_content = self._merge_content(ctx, survivor_of, kept)

    def _first_match(
        self, vector: list[float], kept_vectors: list[list[float]]
    ) -> int | None:
        """Index of the first kept representative within ``threshold`` of
        ``vector``, or ``None``. First-match (not best-match) keeps the pass a
        single linear scan and is deterministic given the stable concept order.
        """
        for index, other in enumerate(kept_vectors):
            if _cosine(vector, other) >= self._threshold:
                return index
        return None

    def _merge_content(
        self,
        ctx: IngestionContext,
        survivor_of: dict[str, str],
        kept: list[Concept],
    ) -> list[ConceptContent]:
        """Rebuild the content list for survivors only, unioning the source_refs
        of every merged duplicate onto the representative's content so the prose
        record stays as well-grounded as the structure record."""
        kept_ids = {c.id for c in kept}
        rebuilt: dict[str, ConceptContent] = {}
        for content in ctx.concept_content:
            rep_id = survivor_of.get(content.concept_id, content.concept_id)
            if rep_id not in kept_ids:
                continue
            if rep_id not in rebuilt:
                # First content for this representative: re-key it to the
                # representative (handles the case where the rep itself was the
                # first seen, so rep_id == content.concept_id).
                rebuilt[rep_id] = replace(content, concept_id=rep_id)
            else:
                existing = rebuilt[rep_id]
                rebuilt[rep_id] = replace(
                    existing,
                    source_refs=_union_refs(existing.source_refs, content.source_refs),
                )
        # Preserve representative order.
        return [rebuilt[c.id] for c in kept if c.id in rebuilt]

    def _text_for(self, ctx: IngestionContext, concept: Concept) -> str:
        """The text embedded for similarity: the body when present, else the
        title plus description. The body is the richest signal for whether two
        concepts are the same."""
        content = ctx.concept_content_for(concept.id)
        if content is not None and content.body:
            return content.body
        return f"{concept.title} {concept.description}".strip()


# --------------------------------------------------------------------------- #
# Pass 3: infer typed edges between the surviving concepts.
# --------------------------------------------------------------------------- #
_EDGES_SYSTEM = (
    "You infer typed edges (prerequisite|encompasses|related) between concepts. "
    "Cite each edge with a source_ref; never invent one. Reply with JSON only."
)


class InferEdgesPass(IngestionPass):
    """Prompt the LLM for prerequisite/encompasses/related edges.

    The JSON contract is::

        {"edges": [{"src": "id", "dst": "id", "type": "prerequisite",
                    "weight": 0.0-1.0, "importance": 0.0-1.0,
                    "rationale": "...", "source_ref": {"file": "...", "line": int}}]}

    Malformed edges (missing endpoints, unknown type, self-loops) are skipped
    here; dangling references to non-existent concepts are tolerated and left
    for VerifyPass to prune once the final concept set is known, so this pass
    stays a pure structural translator.
    """

    name = "infer_edges"

    def __init__(self, llm: LlmProvider, *, batch_size: int = 30) -> None:
        self._llm = llm
        self._batch_size = batch_size

    def run(self, ctx: IngestionContext) -> None:
        concepts = ctx.concepts
        if not concepts:
            return
        files = ctx.source_files()
        # One call per batch of <=batch_size SOURCE concepts (the full id set is
        # supplied for valid endpoints). Bounding the output is what avoids the
        # JSON truncation that silently dropped ALL edges on high-concept decks.
        bs = self._batch_size
        for i in range(0, len(concepts), bs):
            self._infer(ctx, concepts[i : i + bs], concepts, files)

    def _infer(
        self,
        ctx: IngestionContext,
        focus: Sequence[Concept],
        all_concepts: Sequence[Concept],
        files: set[str],
    ) -> None:
        raw = self._llm.complete(
            self._prompt(focus, all_concepts, files),
            system=_EDGES_SYSTEM,
            temperature=0.0,
        )
        for item in _items(raw, "edges"):
            edge = self._build(item)
            if edge is not None:
                ctx.edges.append(edge)

    def _prompt(
        self,
        focus: Sequence[Concept],
        all_concepts: Sequence[Concept],
        source_files: set[str],
    ) -> str:
        """Edge-inference prompt: find edges ORIGINATING FROM the focus concepts,
        with the full id set supplied as valid endpoints. Bounding the source set
        bounds the output (avoids JSON truncation on big decks). An edge must cite
        a real input file (the grounding gate drops edges citing an unknown file),
        so the legitimate files are listed -- the same discipline the extract pass
        gets implicitly from its chunk."""
        listing = "\n".join(
            f"- {c.id}: {c.title} -- {c.description}" for c in focus
        )
        all_ids = ", ".join(c.id for c in all_concepts)
        files = ", ".join(sorted(source_files)) or "(none)"
        return (
            "Infer edges for the following concepts (use ONLY the exact concept "
            "ids -- copy them verbatim; never abbreviate or invent an id):\n"
            f"{listing}\n\n"
            f"Valid endpoint ids (src and dst MUST come from this set): {all_ids}\n\n"
            "For each relationship return src, dst, type "
            "(prerequisite|encompasses|related), weight (0..1, for encompasses), "
            "importance (0..1), rationale, and source_ref {file, line} where "
            f"file MUST be one of these source files: {files}.\n"
            'Return JSON: {"edges": [...]}.'
        )

    def _build(self, item: dict[str, Any]) -> Edge | None:
        """Build one :class:`Edge` from a JSON object, or ``None`` if unusable."""
        src = str(item.get("src") or "").strip()
        dst = str(item.get("dst") or "").strip()
        if not src or not dst or src == dst:
            return None  # endpoints required; a self-loop carries no information
        try:
            edge_type = EdgeType(str(item.get("type") or "").strip().lower())
        except ValueError:
            return None  # unknown relationship kind -> drop rather than guess
        weight = _clamp(_as_float(item.get("weight"), 1.0), 0.0, 1.0)
        importance = _clamp(_as_float(item.get("importance"), 0.5), 0.0, 1.0)
        rationale = item.get("rationale")
        rationale = str(rationale).strip() if rationale else None
        return Edge(
            src=src,
            dst=dst,
            type=edge_type,
            weight=weight,
            importance=importance,
            rationale=rationale,
            source_ref=_to_source_ref(item.get("source_ref")),
        )


# --------------------------------------------------------------------------- #
# Pass 4: generate questions per concept and per important edge (KNIGHT).
# --------------------------------------------------------------------------- #
_QGEN_SYSTEM = (
    "You write grounded exam questions with a grading rubric. "
    "Reply with JSON only."
)


class QuestionGenPass(IngestionPass):
    """Generate difficulty/hop-graded questions (the KNIGHT pass).

    Two sources of questions:

    * One or more single-concept questions per concept (hop_count 1), graded by
      the model on a 1..5 difficulty scale.
    * Multi-hop questions for each *important* edge -- edges whose
      ``importance`` is at least ``min_edge_importance`` -- so the connections
      that matter for the exam are themselves examinable (hop_count defaults to
      2 and the question is tagged with the ``edge_id``).

    Question prose (prompt + rubric) is accumulated as :class:`QuestionContent`;
    questions inherit their concept's (or edge's) source_refs so they stay
    grounded. Question ids are derived deterministically from the concept/edge
    id plus the answer index, so re-running yields stable ids.
    """

    name = "question_gen"

    def __init__(self, llm: LlmProvider, *, min_edge_importance: float = 0.5) -> None:
        self._llm = llm
        self._min_edge_importance = min_edge_importance

    def run(self, ctx: IngestionContext) -> None:
        for concept in ctx.concepts:
            content = ctx.concept_content_for(concept.id)
            raw = self._llm.complete(
                self._concept_prompt(concept, content),
                system=_QGEN_SYSTEM,
                temperature=0.0,
            )
            for index, item in enumerate(_items(raw, "questions")):
                self._emit(
                    ctx,
                    item,
                    id_prefix=concept.id,
                    concept_id=concept.id,
                    base_refs=concept.source_refs,
                    index=index,
                    default_hop=1,
                    edge_id=None,
                )

        for edge in ctx.edges:
            if edge.importance < self._min_edge_importance:
                continue  # only the exam-relevant connections earn a question
            raw = self._llm.complete(
                self._edge_prompt(edge), system=_QGEN_SYSTEM, temperature=0.0
            )
            base_refs = self._edge_refs(ctx, edge)
            for index, item in enumerate(_items(raw, "questions")):
                self._emit(
                    ctx,
                    item,
                    id_prefix=edge.id,
                    concept_id=edge.src,
                    base_refs=base_refs,
                    index=index,
                    default_hop=2,
                    edge_id=edge.id,
                )

    def _emit(
        self,
        ctx: IngestionContext,
        item: dict[str, Any],
        *,
        id_prefix: str,
        concept_id: str,
        base_refs: tuple[SourceRef, ...],
        index: int,
        default_hop: int,
        edge_id: str | None,
    ) -> None:
        """Build and append one Question + QuestionContent from a JSON object.

        A question with no prompt text is skipped (an empty prompt is not a
        question). Difficulty is clamped to 1..5 and hop_count floored at 1 to
        honour the entity's documented ranges.
        """
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            return
        rubric = str(item.get("rubric") or "").strip()
        kind = str(item.get("kind") or "open").strip() or "open"
        difficulty = int(_clamp(_as_int(item.get("difficulty"), 1), 1, 5))
        hop_count = max(1, _as_int(item.get("hop_count"), default_hop))
        refs = _to_source_refs(item.get("source_refs")) or base_refs
        question_id = f"{id_prefix}::q{index}"
        ctx.questions.append(
            Question(
                id=question_id,
                concept_id=concept_id,
                kind=kind,
                difficulty=difficulty,
                hop_count=hop_count,
                edge_id=edge_id,
                source_refs=refs,
                generated_by=_GENERATED_BY,
            )
        )
        ctx.question_content.append(
            QuestionContent(question_id=question_id, prompt=prompt, rubric=rubric)
        )

    def _edge_refs(self, ctx: IngestionContext, edge: Edge) -> tuple[SourceRef, ...]:
        """Grounding refs for an edge question: the edge's own citation if it
        has one, else the source concept's refs (a multi-hop question is still
        anchored at its source concept)."""
        if edge.source_ref is not None:
            return (edge.source_ref,)
        source = next((c for c in ctx.concepts if c.id == edge.src), None)
        return source.source_refs if source is not None else ()

    def _concept_prompt(
        self, concept: Concept, content: ConceptContent | None
    ) -> str:
        """Build the single-concept question prompt."""
        body = content.body if content is not None else ""
        return (
            f"Generate exam questions for concept {concept.id}.\n"
            f"Title: {concept.title}\n"
            f"Description: {concept.description}\n"
            f"Body: {body}\n\n"
            'Return JSON: {"questions": [{"kind": "open|mcq|derivation", '
            '"difficulty": 1-5, "prompt": "...", "rubric": "...", '
            '"source_refs": [{"file": "...", "line": int}]}]}.'
        )

    def _edge_prompt(self, edge: Edge) -> str:
        """Build the multi-hop edge question prompt."""
        return (
            f"Generate a multi-hop question for edge {edge.id}.\n"
            f"This {edge.type.value} edge connects {edge.src} -> {edge.dst}.\n"
            f"Rationale: {edge.rationale}\n\n"
            'Return JSON: {"questions": [{"kind": "open", "difficulty": 1-5, '
            '"hop_count": 2, "prompt": "...", "rubric": "..."}]}.'
        )


# --------------------------------------------------------------------------- #
# Pass 5: adversarial grounding gate.
# --------------------------------------------------------------------------- #
class VerifyPass(IngestionPass):
    """Drop anything that cannot be traced to a real source (the grounding gate).

    This is the anti-fabrication backstop. It runs last and enforces three
    invariants on the working set before persistence:

    * A concept survives only if it has at least one source_ref whose file was
      actually fed to this ingest run. Empty refs and hallucinated files (a
      citation to a file that was never an input) are both rejected.
    * An edge survives only if it is itself grounded AND both of its endpoints
      survived (no dangling edges into dropped concepts).
    * A question survives only if its concept survived AND -- when it is an
      edge/multi-hop question -- its edge survived (no orphan questions).

    Content records for dropped concepts/questions are pruned in lockstep so
    persistence never writes prose with no owning structure.
    """

    name = "verify"

    def run(self, ctx: IngestionContext) -> None:
        files = ctx.source_files()

        kept_concepts = [c for c in ctx.concepts if self._grounded(c.source_refs, files)]
        kept_ids = {c.id for c in kept_concepts}

        kept_edges = [
            e
            for e in ctx.edges
            if self._edge_grounded(e, files)
            and e.src in kept_ids
            and e.dst in kept_ids
        ]
        kept_edge_ids = {e.id for e in kept_edges}

        kept_questions = [
            q
            for q in ctx.questions
            if q.concept_id in kept_ids
            and (q.edge_id is None or q.edge_id in kept_edge_ids)
        ]
        kept_question_ids = {q.id for q in kept_questions}

        ctx.concept_content = [
            cc for cc in ctx.concept_content if cc.concept_id in kept_ids
        ]
        ctx.question_content = [
            qc for qc in ctx.question_content if qc.question_id in kept_question_ids
        ]
        ctx.concepts = kept_concepts
        ctx.edges = kept_edges
        ctx.questions = kept_questions

    @staticmethod
    def _grounded(refs: Sequence[SourceRef], files: set[str]) -> bool:
        """True iff at least one ref points at a real input file."""
        return any(ref.file in files for ref in refs)

    @classmethod
    def _edge_grounded(cls, edge: Edge, files: set[str]) -> bool:
        """True iff the edge carries a citation to a real input file."""
        if edge.source_ref is None:
            return False
        return edge.source_ref.file in files
