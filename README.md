# Curriculum engine

A knowledge-graph engine that decides **what to teach or test next, and in what order**,
so a tutor LLM never has to make curriculum decisions mid-session. The model becomes the
*delivery* layer; this is the *curriculum* layer. You point it at your own plain-text
course materials; it builds a concept/edge graph, generates grounded exam questions, and
serves the result to a host (Hermes) over MCP.

No source materials and no secrets are committed here -- you supply your own materials via
`corpus.json` and an OpenAI-compatible provider API key at runtime.

## Quickstart

Full bootstrap (Docker + database, install, build, serve) lives in **[AGENTS.md](AGENTS.md)**.
One screen:

```sh
docker compose up -d db                                   # Postgres+pgvector on :5433
uv venv && uv pip install -e '.[postgres,mcp,nous]'       # install with all adapters
export CURRICULUM_API_KEY=...                              # provider API key
export CURRICULUM_BASE_URL=https://inference-api.nousresearch.com/v1  # any OpenAI-compatible base URL
cp corpus.example.json corpus.json                         # then edit it for your course
.venv/bin/curriculum build corpus.json                     # ingest -> link -> questions
.venv/bin/curriculum status --course <your-course>         # read-only graph counts
```

Run `.venv/bin/curriculum doctor` first to check prerequisites (docker, DB, key, bundle).
Tests need neither a database nor a key: `make test`.


## Provider configuration

The build pipeline calls an OpenAI-compatible provider for chat completions and
embeddings. Nous remains the default base URL for backwards compatibility, but
you can point `CURRICULUM_BASE_URL` at any compatible vendor. Legacy
`NOUS_API_KEY` and `NOUS_BASE_URL` still work as fallbacks.

```sh
# Generic OpenAI-compatible configuration
export CURRICULUM_API_KEY=...
export CURRICULUM_BASE_URL=https://inference-api.nousresearch.com/v1
export CURRICULUM_INGEST_MODEL=deepseek/deepseek-v4-flash
export CURRICULUM_EMBED_MODEL=google/gemini-embedding-2
export CURRICULUM_EMBED_DIM=3072

# NVIDIA NIM-style example (check NVIDIA's current catalog for exact model ids)
export CURRICULUM_API_KEY="$NVIDIA_API_KEY"
export CURRICULUM_BASE_URL=https://integrate.api.nvidia.com/v1
export CURRICULUM_INGEST_MODEL=<nvidia-chat-model>
export CURRICULUM_EMBED_MODEL=<nvidia-embedding-model>
export CURRICULUM_EMBED_DIM=<embedding-dimension>
```

`CURRICULUM_EMBED_DIM` must match both the embedding model output dimension and
the `vector(N)` column in `schema/001_init.sql`; if you change embedding
dimensions, update the schema and recreate/migrate the database.

## Architecture (ports & adapters / hexagonal)

```
domain/        immutable entities + DTOs + events (no I/O, no deps)
ports/         abstract interfaces: repositories, strategies, providers, service
engine/        FSRS scheduler, FIRe propagation, scoring terms, selection policy
storage/       InMemory + Postgres repositories, OKF content repository
okf/           OKF v0.1 (de)serialization
ingestion/     multipass pipeline: extract -> dedupe -> infer-edges -> verify
linking/       embedding-guided edge repair for isolated concepts
archetypes/    course strategy templates (conceptual-written, procedural, ...)
sync/          OKF <-> Postgres reconciliation (hash-keyed, one-way)
application/    the CurriculumService use-cases (next/explain/quiz/grade/state)
app/           config-driven build orchestration (ingest/link/questions/status)
cli/           the agent-facing `curriculum` entrypoint
mcp/           stdio MCP server exposing the service to Hermes
```

The core depends only on `domain` + `ports`; everything external is a swappable adapter
(Dependency Inversion). The engine therefore runs and tests with in-memory + fake adapters,
with no Postgres and no paid inference. Heavy/optional imports (`psycopg`, `mcp`, provider clients) are deferred into the functions
that use them, so the package imports cleanly on a machine that has none of them installed.

## Storage split (polyglot, single-ownership)

- **OKF markdown bundle** owns the *content*: concept/question prose and citations.
  Human-readable, git-diffable, hand-editable; the source of truth for text.
- **Postgres + pgvector** owns the *structure + metadata + state*: edges, importance,
  skip-counts, embeddings, and all per-learner FSRS/review state.
- A one-way, content-hash-keyed **sync** reconciles OKF -> Postgres (and re-embeds changed
  content), so no fact lives authoritatively in two places and nothing drifts.

## The engine

- **FSRS** -- the spaced-repetition scheduler (when each concept is next due).
- **FIRe** -- importance propagation across the edge graph after a graded answer.
- **Selection** -- a weighted-sampling policy over five scoring terms (urgency, difficulty
  fit, exploration, interleave penalty, coverage) that picks the single best next action and
  exposes the full ranked field.

These feed the five MCP tools -- `next`, `explain`, `quiz`, `grade`, `state` -- which are
thin adapters over the same `CurriculumService` use-cases the unit tests drive directly.

## Design docs

- [`docs/okf-spec.md`](docs/okf-spec.md) -- the OKF v0.1 content-bundle format (frontmatter,
  ids, citations) that the file content repository reads and writes.

## Build & serve

See **[AGENTS.md](AGENTS.md)** for the full runbook (prerequisites, env vars, reset, and
troubleshooting). The `Makefile` wraps the common steps: `make db`, `make install`,
`make build`, `make link`, `make questions`, `make status`, `make mcp`, `make reset`,
`make test`.
