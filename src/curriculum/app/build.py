"""Config-driven build orchestration: run the whole corpus flow from a manifest.

This module is the reproducible replacement for the ad-hoc ``scripts/`` one-offs
(``ingest_corpus.py`` + ``gen_questions.py``): instead of hardcoded file lists,
absolute bundle paths, and a key scraped out of one user's ``~/.hermes`` config,
every collaborator is constructed from a :class:`curriculum.config.Settings` and
every input comes from a declarative *manifest* dict. That makes the build
agent-bootstrappable -- point a manifest at any plain-text course materials, hand
in a Settings, and the same code path produces the graph, links the isolated
concepts, and generates questions.

Why the heavy imports are deferred
----------------------------------
The Postgres adapter (``psycopg``/``pgvector``) and the embedding linker are
optional, environment-specific dependencies. They are imported *inside* the
functions that need them, never at module load, so ``import curriculum.app.build``
succeeds on a machine with no database driver and no network -- which is exactly
what lets the manifest loader (pure JSON + validation) be unit-tested offline and
lets the rest of the test-suite import this module freely. The OpenAI-compatible
provider adapter and the ingestion passes, by contrast, are stdlib-only and safe
to import eagerly.

Standard library plus the project's own modules only.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..config import Settings
from ..domain.entities import Question, QuestionContent, SourceRef
from ..domain.errors import ConfigError
from ..ingestion.passes import (
    DedupePass,
    ExtractPass,
    InferEdgesPass,
    IngestionContext,
    VerifyPass,
)
from ..ingestion.pipeline import Pipeline
from ..providers_openai_compatible import OpenAICompatibleEmbedder, OpenAICompatibleLlm
from ..storage.okf_content import FileContentRepository

__all__ = [
    "load_manifest",
    "ingest",
    "link",
    "generate_questions",
    "status",
]

# Defaults that the scripts hardcoded; kept here as named constants so the WHY of
# each magic number is documented in one place rather than scattered inline.
_DEFAULT_CHUNK_LINES = 150  # big chunks => few extract calls (batched extraction)
_DEFAULT_WORKERS = 6        # concurrent sources/batches; network-bound, so > CPU count
_CONCEPT_BATCH = 12         # concepts per question-gen LLM call (not one call per concept)
_EDGE_BATCH = 10            # important edges per multi-hop question-gen call
_MIN_EDGE_IMPORTANCE = 0.5  # only exam-relevant connections earn a multi-hop question
_QGEN_SYSTEM = (
    "You write grounded exam questions with a concise grading rubric. "
    "Reply with JSON only."
)
# Provenance stamped on every generated Question so its origin (this module) is
# auditable in the persisted row, exactly as the ingestion passes tag theirs.
_GENERATED_BY = "app.build.generate_questions"


# --------------------------------------------------------------------------- #
# Manifest loading + validation (pure: no I/O beyond reading the file).
# --------------------------------------------------------------------------- #
def load_manifest(path: str) -> dict:
    """Load and validate a corpus manifest, returning a normalised dict.

    The manifest declares one course's source materials (see
    ``corpus.example.json``)::

        {"course": "...", "chunk_lines": 150,
         "sources": [{"path": "...", "token": "..."}, ...]}

    ``course`` (non-empty string) and ``sources`` (non-empty list of
    ``{path, token}`` objects) are required; ``chunk_lines`` defaults to
    :data:`_DEFAULT_CHUNK_LINES` when absent. Validation is strict and fail-fast
    because a malformed manifest would otherwise surface much later as an opaque
    error mid-ingest (after paid inference has already been spent); any defect
    raises :class:`curriculum.domain.errors.ConfigError` with a pointed message.
    The returned dict is normalised -- ``sources`` carries exactly ``path`` and
    ``token`` per entry -- so downstream code can trust the shape without
    re-checking it.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read manifest {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"manifest {path!r} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("manifest must be a JSON object")

    course = data.get("course")
    if not isinstance(course, str) or not course.strip():
        raise ConfigError("manifest 'course' must be a non-empty string")

    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError(
            "manifest 'sources' must be a non-empty list of {path, token} objects"
        )

    normalised_sources: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ConfigError(f"manifest source #{index} must be an object")
        source_path = source.get("path")
        token = source.get("token")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ConfigError(
                f"manifest source #{index} must have a non-empty 'path'"
            )
        if not isinstance(token, str) or not token.strip():
            raise ConfigError(
                f"manifest source #{index} must have a non-empty 'token'"
            )
        # Keep only the two fields the build consumes; drop any extras (e.g. a
        # human-facing comment) so the normalised shape is exact.
        normalised_sources.append({"path": source_path, "token": token})

    chunk_lines = data.get("chunk_lines", _DEFAULT_CHUNK_LINES)
    # bool is an int subclass; reject it explicitly so ``true`` is not read as 1.
    if isinstance(chunk_lines, bool) or not isinstance(chunk_lines, int) or chunk_lines <= 0:
        raise ConfigError("manifest 'chunk_lines' must be a positive integer")

    return {
        "course": course,
        "chunk_lines": chunk_lines,
        "sources": normalised_sources,
    }


# --------------------------------------------------------------------------- #
# Stage 1: ingest sources into the concept/edge graph (graph-only pipeline).
# --------------------------------------------------------------------------- #
def ingest(manifest: dict, settings: Settings) -> dict:
    """Ingest every manifest source into the knowledge graph, concurrently.

    Each source is read, split into ``chunk_lines``-line chunks (each tagged with
    the source's stable ``token`` as its grounding citation and the 1-based start
    line), and run through a GRAPH-ONLY pipeline --
    ``Pipeline([ExtractPass, DedupePass, InferEdgesPass, VerifyPass])``. Question
    generation is deliberately NOT part of this pipeline (it is the separate,
    batched :func:`generate_questions` follow-up), which keeps each source to
    roughly one extract call, one edge call, and one embedding call.

    Sources are processed on a thread pool (network-bound work) with each worker
    opening its OWN Postgres connection -- psycopg connections are not thread
    safe, and the autocommit connection commits each source independently. That
    independence is also why a single failing source is tolerated rather than
    aborting the batch: the inference already spent on the other sources, and
    their committed rows, must not be thrown away. ``files`` therefore counts the
    sources that succeeded; a caller can compare it against ``len(sources)`` to
    detect a partial run.

    Returns aggregate counts ``{files, concepts, edges}``.
    """
    _require_api_key(settings)
    # The OKF content repository is stateless (just a root path) and writes one
    # file per concept id, so a single instance is safely shared across workers.
    content = FileContentRepository(Path(settings.okf_bundle_path))
    sources = manifest["sources"]
    workers = _worker_count(settings, len(sources))

    files = concepts = edges = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_ingest_source, source, manifest, settings, content): source
            for source in sources
        }
        for future in as_completed(futures):
            try:
                counts = future.result()
            except Exception:  # noqa: BLE001 - one bad source must not sink the batch
                continue
            files += 1
            concepts += counts["concepts"]
            edges += counts["edges"]

    return {"files": files, "concepts": concepts, "edges": edges}


def _ingest_source(
    source: dict, manifest: dict, settings: Settings, content: FileContentRepository
) -> dict:
    """Ingest one source file in its own thread/connection (see :func:`ingest`)."""
    # Local import keeps the Postgres driver optional at module-load time.
    from ..storage.postgres import PostgresRepositories, connect

    token = source["token"]
    lines = Path(source["path"]).read_text(encoding="utf-8", errors="ignore").splitlines()
    chunk_lines = manifest["chunk_lines"]
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(lines), chunk_lines):
        text = "\n".join(lines[start : start + chunk_lines]).strip()
        if text:
            # ``file`` is the stable token (the grounding citation), ``line`` the
            # 1-based start line so a citation can point back into the source.
            chunks.append({"text": text, "file": token, "line": start + 1})

    llm = _make_llm(settings)
    embedder = _make_embedder(settings)
    context = IngestionContext(course=manifest["course"], chunks=chunks)
    pipeline = Pipeline(
        [ExtractPass(llm), DedupePass(embedder), InferEdgesPass(llm), VerifyPass()]
    )
    pipeline.run(context)

    connection = connect(settings.database_url)
    try:
        repos = PostgresRepositories(connection)
        counts = pipeline.persist(
            context,
            concepts=repos.concepts,
            edges=repos.edges,
            questions=repos.questions,
            content=content,
        )
    finally:
        connection.close()
    return {"concepts": counts["concepts"], "edges": counts["edges"]}


# --------------------------------------------------------------------------- #
# Stage 2: link the isolated concepts (embedding-guided edge repair).
# --------------------------------------------------------------------------- #
def link(settings: Settings, course: str) -> dict:
    """Connect isolated concepts to the graph via embedding-guided edge repair.

    Delegates to the :class:`EmbeddingLinker`, which (for each concept that has
    no edges) retrieves its nearest neighbours by vector similarity and asks the
    LLM only to CLASSIFY the edge type among those few candidates -- a small,
    reliable prompt rather than a cold guess across the whole id space. This is
    the standard post-ingest step, not a manual patch, so it lives behind the
    same Settings-driven seam as the rest of the build.

    Returns whatever counts the linker reports (e.g. inferred/persisted/still
    isolated) as a plain dict.
    """
    _require_api_key(settings)
    # Both imports are deferred: the linker is the refactor of repair_emb.py and
    # the Postgres adapter is optional, so this module imports without either.
    from ..linking.embedding_linker import EmbeddingLinker
    from ..storage.postgres import PostgresRepositories, connect

    connection = connect(settings.database_url)
    try:
        repos = PostgresRepositories(connection)
        linker = EmbeddingLinker(
            repos.concepts,
            repos.edges,
            _make_llm(settings),
        )
        return dict(linker.link_isolated(course))
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Stage 3: batched KNIGHT question generation over the persisted graph.
# --------------------------------------------------------------------------- #
def generate_questions(settings: Settings, course: str) -> dict:
    """Generate exam questions over the persisted graph, in batches, threaded.

    Standalone from the per-item ``QuestionGenPass`` on purpose: it reads the
    already-persisted concepts (plus their OKF bodies) and important edges for
    ``course`` from Postgres, then generates questions in BATCHES -- one LLM call
    per ~12 concepts and per ~10 edges, not one call per item -- which is what
    keeps generation cheap on a large deck. Generation runs on a thread pool (the
    LLM calls are independent and network-bound); persistence then runs serially
    on the single main connection (psycopg connections are not thread safe).
    Question ids are derived deterministically from the concept/edge id, so the
    write is idempotent and a re-run overwrites rather than duplicates.

    Returns ``{questions: int}`` -- the number of questions generated/persisted.
    """
    _require_api_key(settings)
    from ..storage.postgres import PostgresRepositories, connect

    content = FileContentRepository(Path(settings.okf_bundle_path))
    connection = connect(settings.database_url)
    try:
        repos = PostgresRepositories(connection)
        concept_batches, refs_by_id = _concept_batches(connection, content, course)
        edge_batches = _edge_batches(connection, course)
        workers = _worker_count(settings, len(concept_batches) + len(edge_batches))

        generated: list[tuple[Question, QuestionContent]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_gen_concept_batch, settings, batch, refs_by_id)
                for batch in concept_batches
            ]
            futures += [
                executor.submit(_gen_edge_batch, settings, batch, refs_by_id)
                for batch in edge_batches
            ]
            for future in as_completed(futures):
                generated.extend(future.result())

        # Persist on the main connection: content first, then the index row.
        for question, question_content in generated:
            content.put_question_content(question_content)
            repos.questions.upsert(question)
    finally:
        connection.close()

    return {"questions": len(generated)}


def _concept_batches(
    connection, content: FileContentRepository, course: str
) -> tuple[list[list[tuple]], dict[str, tuple[SourceRef, ...]]]:
    """Load the course's concepts (+ OKF bodies) into question-gen batches.

    Returns the batches (each a list of ``(id, title, description, body)``) and a
    ``concept_id -> source_refs`` map so generated questions inherit their
    concept's grounding citations.
    """
    refs_by_id: dict[str, tuple[SourceRef, ...]] = {}
    batches: list[list[tuple]] = []
    current: list[tuple] = []
    rows = connection.execute(
        "SELECT id, title, description, source_refs FROM concept "
        "WHERE course = %s ORDER BY id",
        (course,),
    ).fetchall()
    for concept_id, title, description, refs in rows:
        refs_by_id[concept_id] = tuple(
            SourceRef(ref["file"], ref.get("line"))
            for ref in (refs or [])
            if ref.get("file")
        )
        concept_content = content.get_concept_content(concept_id)
        body = concept_content.body if concept_content is not None else ""
        current.append((concept_id, title, description or "", body))
        if len(current) >= _CONCEPT_BATCH:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches, refs_by_id


def _edge_batches(connection, course: str) -> list[list[tuple]]:
    """Load the course's *important* edges into multi-hop question-gen batches.

    Edges carry no course column, so scope through the SOURCE concept (the graph
    is intra-course and directed). Each batch entry is
    ``(edge_id, src, dst, type, rationale)``.
    """
    batches: list[list[tuple]] = []
    current: list[tuple] = []
    rows = connection.execute(
        "SELECT e.src, e.dst, e.type, e.rationale "
        "FROM edge e JOIN concept c ON e.src = c.id "
        "WHERE c.course = %s AND e.importance >= %s ORDER BY e.src",
        (course, _MIN_EDGE_IMPORTANCE),
    ).fetchall()
    for src, dst, edge_type, rationale in rows:
        current.append((f"{src}::{edge_type}::{dst}", src, dst, edge_type, rationale or ""))
        if len(current) >= _EDGE_BATCH:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _gen_concept_batch(
    settings: Settings, batch: list[tuple], refs_by_id: dict[str, tuple[SourceRef, ...]]
) -> list[tuple[Question, QuestionContent]]:
    """Generate 1-2 single-concept questions for each concept in one LLM call."""
    blocks = [
        f"- concept_id: {concept_id}\n  title: {title}\n  description: {description}"
        f"\n  body: {body[:400]}"
        for concept_id, title, description, body in batch
    ]
    prompt = (
        "Generate 1-2 exam questions for EACH concept below. Return JSON "
        '{"questions": [{"concept_id": "<exact id from the list>", "kind": '
        '"open|mcq|derivation", "difficulty": 1-5, "prompt": "...", "rubric": "..."}]}. '
        "Every question MUST carry the exact concept_id it tests.\n\n" + "\n".join(blocks)
    )
    llm = _make_llm(settings)
    raw = llm.complete(prompt, system=_QGEN_SYSTEM, temperature=0.0)

    valid_ids = {row[0] for row in batch}
    counters: dict[str, int] = {}
    out: list[tuple[Question, QuestionContent]] = []
    for item in _items(raw, "questions"):
        concept_id = str(item.get("concept_id") or "").strip()
        prompt_text = str(item.get("prompt") or "").strip()
        if concept_id not in valid_ids or not prompt_text:
            continue  # drop hallucinated ids / empty prompts (anti-fabrication)
        index = counters.get(concept_id, 0)
        counters[concept_id] = index + 1
        question_id = f"{concept_id}::q{index}"
        out.append(
            (
                Question(
                    id=question_id,
                    concept_id=concept_id,
                    kind=str(item.get("kind") or "open").strip() or "open",
                    difficulty=_clamp_int(item.get("difficulty"), 1, 5, 1),
                    hop_count=1,
                    edge_id=None,
                    source_refs=refs_by_id.get(concept_id, ()),
                    generated_by=_GENERATED_BY,
                ),
                QuestionContent(
                    question_id=question_id,
                    prompt=prompt_text,
                    rubric=str(item.get("rubric") or "").strip(),
                ),
            )
        )
    return out


def _gen_edge_batch(
    settings: Settings, batch: list[tuple], refs_by_id: dict[str, tuple[SourceRef, ...]]
) -> list[tuple[Question, QuestionContent]]:
    """Generate one multi-hop question per edge in one LLM call (hop_count >= 2)."""
    blocks = [
        f"- edge_id: {edge_id}\n  {src} --{edge_type}--> {dst}\n  rationale: {rationale}"
        for edge_id, src, dst, edge_type, rationale in batch
    ]
    prompt = (
        "Generate ONE multi-hop exam question for EACH edge below (it must require "
        'connecting BOTH concepts). Return JSON {"questions": [{"edge_id": '
        '"<exact edge_id>", "kind": "open", "difficulty": 1-5, "hop_count": 2, '
        '"prompt": "...", "rubric": "..."}]}.\n\n' + "\n".join(blocks)
    )
    llm = _make_llm(settings)
    raw = llm.complete(prompt, system=_QGEN_SYSTEM, temperature=0.0)

    by_edge_id = {row[0]: row for row in batch}
    counters: dict[str, int] = {}
    out: list[tuple[Question, QuestionContent]] = []
    for item in _items(raw, "questions"):
        edge_id = str(item.get("edge_id") or "").strip()
        prompt_text = str(item.get("prompt") or "").strip()
        if edge_id not in by_edge_id or not prompt_text:
            continue
        src = by_edge_id[edge_id][1]
        index = counters.get(edge_id, 0)
        counters[edge_id] = index + 1
        question_id = f"{edge_id}::q{index}"
        out.append(
            (
                Question(
                    id=question_id,
                    concept_id=src,  # anchor a multi-hop question at its source concept
                    kind="open",
                    difficulty=_clamp_int(item.get("difficulty"), 1, 5, 3),
                    hop_count=max(1, _clamp_int(item.get("hop_count"), 1, 9, 2)),
                    edge_id=edge_id,
                    source_refs=refs_by_id.get(src, ()),
                    generated_by=_GENERATED_BY,
                ),
                QuestionContent(
                    question_id=question_id,
                    prompt=prompt_text,
                    rubric=str(item.get("rubric") or "").strip(),
                ),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 4: status report (counts for the course).
# --------------------------------------------------------------------------- #
def status(settings: Settings, course: str) -> dict:
    """Report graph counts for ``course``: concepts, edges, questions, isolated.

    A cheap, read-only summary (no inference) used to confirm a build landed and
    to drive the linker decision (how many concepts are still isolated). Edges
    and questions are scoped to the course through the SOURCE/owning concept,
    since neither table carries a course column.
    """
    from ..storage.postgres import connect

    connection = connect(settings.database_url)
    try:
        concepts = connection.execute(
            "SELECT count(*) FROM concept WHERE course = %s", (course,)
        ).fetchone()[0]
        edges = connection.execute(
            "SELECT count(*) FROM edge e JOIN concept c ON e.src = c.id "
            "WHERE c.course = %s",
            (course,),
        ).fetchone()[0]
        questions = connection.execute(
            "SELECT count(*) FROM question q JOIN concept c ON q.concept_id = c.id "
            "WHERE c.course = %s",
            (course,),
        ).fetchone()[0]
        isolated = connection.execute(
            "SELECT count(*) FROM concept c WHERE c.course = %s AND NOT EXISTS "
            "(SELECT 1 FROM edge e WHERE e.src = c.id OR e.dst = c.id)",
            (course,),
        ).fetchone()[0]
    finally:
        connection.close()

    return {
        "concepts": concepts,
        "edges": edges,
        "questions": questions,
        "isolated": isolated,
    }


# --------------------------------------------------------------------------- #
# Small shared helpers.
# --------------------------------------------------------------------------- #
def _make_llm(settings: Settings) -> OpenAICompatibleLlm:
    """Construct the configured OpenAI-compatible chat provider."""
    return OpenAICompatibleLlm(
        api_key=settings.api_key or "",
        base_url=settings.base_url,
        model=settings.ingest_model,
    )


def _make_embedder(settings: Settings) -> OpenAICompatibleEmbedder:
    """Construct the configured OpenAI-compatible embeddings provider."""
    return OpenAICompatibleEmbedder(
        api_key=settings.api_key or "",
        base_url=settings.base_url,
        model=settings.embed_model,
        dim=settings.embedding_dim,
    )


def _require_api_key(settings: Settings) -> None:
    """Fail early when no OpenAI-compatible provider API key is configured."""
    if not settings.api_key:
        raise ConfigError(
            "CURRICULUM_API_KEY is not set; OpenAI-compatible build steps need "
            "an API key. Legacy NOUS_API_KEY is also accepted as a fallback."
        )


def _worker_count(settings: Settings, work_items: int) -> int:
    """Choose a thread-pool size: the configured workers, bounded by the work.

    ``Settings`` does not (yet) declare a workers field, so this reads it
    defensively via ``getattr`` and falls back to :data:`_DEFAULT_WORKERS`;
    pinning the pool to at most ``work_items`` avoids spinning idle threads, and
    the ``max(1, ...)`` floor keeps ``ThreadPoolExecutor`` happy on an empty set.
    """
    configured = getattr(settings, "max_workers", None) or _DEFAULT_WORKERS
    return max(1, min(configured, work_items))


def _loads_lenient(text: str) -> Any:
    """Parse JSON that may be wrapped in prose/markdown, ``None`` on failure.

    Mirrors the ingestion pass's lenient parser (kept local so this batched
    generator stays standalone from the per-item pass): try a strict parse, then
    fall back to slicing the outermost ``{...}``/``[...]`` span.
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
    """Extract a list of item dicts from an LLM completion (bare array or wrapped
    under ``key``); anything unparseable yields an empty list -- best effort,
    never an exception."""
    obj = _loads_lenient(text)
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        return [item for item in obj[key] if isinstance(item, dict)]
    return []


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    """Coerce ``value`` to an int clamped to ``[low, high]``; ``default`` on a
    non-numeric value, so one malformed field never aborts a whole batch."""
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default
