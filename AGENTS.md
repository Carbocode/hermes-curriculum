# AGENTS.md -- agent runbook for the curriculum engine

This repository is a self-contained, hexagonal (ports-and-adapters) knowledge-graph
engine that decides **what a learner should be taught or tested on next, and in what
order**, so a tutor LLM never has to make curriculum decisions mid-session. You point
it at your own plain-text course materials; it ingests them into a concept/edge graph
(Postgres + pgvector for structure/state, an OKF markdown bundle for prose), links the
isolated concepts, generates grounded exam questions, and serves the result to a host
(Hermes) over an MCP server exposing `next` / `explain` / `quiz` / `grade` / `state`.
Everything here is generic: **no source materials and no secrets are committed** -- you
supply both at runtime through `corpus.json` and `CURRICULUM_API_KEY` (or legacy `NOUS_API_KEY`).

## Prerequisites

- **Docker + Docker Compose** -- runs the bundled Postgres 16 + pgvector database.
- **uv** (https://docs.astral.sh/uv/) -- creates the virtualenv and installs the package.
- **Python 3.11+** -- the package targets `>=3.11`.
- **An OpenAI-compatible provider API key** -- every build stage (extraction, edge inference, linking,
  question generation, embeddings) calls a provider API. Export it as `CURRICULUM_API_KEY`;
  legacy `NOUS_API_KEY` remains supported as a fallback.
- **Your own plain-text course materials** -- e.g. text extracts of slides/notes. None ship
  with the repo.

Run a readiness check at any time with `.venv/bin/curriculum doctor` (docker, DB, key,
bundle). It exits non-zero if anything is missing, so it doubles as a scriptable gate.

## Setup (run from the repository root, in order)

1. **Start the database.**
   ```sh
   docker compose up -d db
   ```
   Brings up Postgres 16 + pgvector as container `curriculum-db` on host port **5433**
   (chosen to avoid clashing with a local 5432). On first boot `schema/001_init.sql` is
   auto-applied from the init dir, and data persists in the named `curriculum_pg` volume.
   Equivalent once the venv exists: `.venv/bin/curriculum db-up` (and `db-down` to stop).

2. **Create the venv and install the package with all adapters.**
   ```sh
   uv venv && uv pip install -e '.[postgres,mcp,nous]'
   ```
   `postgres` pulls in `psycopg` + `pgvector`, `mcp` the MCP SDK, `nous` the HTTP client.
   The `curriculum` console script lands at `.venv/bin/curriculum`. The core package
   imports without any of these extras (heavy/optional imports are deferred), so
   `curriculum --help` and `curriculum doctor` work even before they install.

3. **Export your API key (and any optional overrides).**
   ```sh
   export CURRICULUM_API_KEY=...  # the only REQUIRED variable
   ```
   `CURRICULUM_API_KEY` is the primary required setting; legacy `NOUS_API_KEY` also works. Everything else has a working default.
   All configuration (`curriculum.config.Settings`, loaded by `curriculum.config.load()`)
   comes from the environment:

   | Variable                  | Default                                                        | Meaning |
   |---------------------------|----------------------------------------------------------------|---------|
   | `CURRICULUM_API_KEY`      | *(unset -- REQUIRED)*                                          | OpenAI-compatible provider API key; legacy `NOUS_API_KEY` fallback supported. Build stages fail fast without it. |
   | `CURRICULUM_DB_URL`       | `postgresql://curriculum:curriculum@localhost:5433/curriculum` | Postgres DSN (matches the docker-compose service). |
   | `CURRICULUM_OKF_PATH`     | `./bundle`                                                      | Directory of the OKF markdown content bundle (created on first ingest). |
   | `CURRICULUM_COURSE`       | `Cybersecurity`                                                | Default course for the `--course`-scoped commands (`status`, `link`, `questions`). |
   | `CURRICULUM_BASE_URL`     | `https://inference-api.nousresearch.com/v1`                    | OpenAI-compatible provider base URL; legacy `NOUS_BASE_URL` fallback supported. |
   | `CURRICULUM_INGEST_MODEL` | `deepseek/deepseek-v4-flash`                                   | LLM for extraction, edge inference, linking, and question generation. |
   | `CURRICULUM_EMBED_MODEL`  | `google/gemini-embedding-2`                                    | Embedding model (3072-dim by default). |
   | `CURRICULUM_EMBED_DIM`    | `3072`                                                         | Embedding dimension. MUST match the `vector(N)` column in `schema/001_init.sql`. |


   Example vendor overrides:
   ```sh
   # Nous/default-compatible setup
   export CURRICULUM_API_KEY="$NOUS_API_KEY"
   export CURRICULUM_BASE_URL="https://inference-api.nousresearch.com/v1"
   export CURRICULUM_INGEST_MODEL="deepseek/deepseek-v4-flash"
   export CURRICULUM_EMBED_MODEL="google/gemini-embedding-2"
   export CURRICULUM_EMBED_DIM=3072

   # NVIDIA NIM-style OpenAI-compatible setup. Check NVIDIA's current catalog
   # for exact chat/embedding model ids and embedding dimensions.
   export CURRICULUM_API_KEY="$NVIDIA_API_KEY"
   export CURRICULUM_BASE_URL="https://integrate.api.nvidia.com/v1"
   export CURRICULUM_INGEST_MODEL="<nvidia-chat-model>"
   export CURRICULUM_EMBED_MODEL="<nvidia-embedding-model>"
   export CURRICULUM_EMBED_DIM="<embedding-dimension>"
   ```
   `CURRICULUM_EMBED_DIM` MUST match both the provider embedding output dimension and
   the `vector(N)` column in `schema/001_init.sql`; if you change dimensions, update
   the schema and recreate/migrate the database.

4. **Create your manifest and point it at your own materials.**
   ```sh
   cp corpus.example.json corpus.json
   ```
   Edit `corpus.json`: set `course` to your course name, and list your plain-text sources
   under `sources` as `{ "path": "...", "token": "..." }` (paths absolute or relative to
   the working dir; `token` is the stable id used as the grounding citation for that
   source). `chunk_lines` (default 150) sets the lines-per-chunk granularity. See
   `corpus.example.json` for the full schema.

5. **Build: ingest -> link -> questions.**
   ```sh
   .venv/bin/curriculum build corpus.json
   ```
   Runs the whole pipeline for the manifest's course and emits each stage's JSON counts as
   it completes. The course for the link/question stages comes from the manifest, not from
   `--course`. To run the stages individually: `curriculum ingest corpus.json`,
   `curriculum link --course <name>`, `curriculum questions --course <name>`.

6. **Check the result.**
   ```sh
   .venv/bin/curriculum status --course <your-course>
   ```
   Prints read-only graph counts (`concepts`, `edges`, `questions`, `isolated`) -- no
   inference, no writes. NOTE: `status` defaults to `CURRICULUM_COURSE` (`Cybersecurity`),
   so pass `--course <your-course>` to match what you just built, or export
   `CURRICULUM_COURSE` first.

7. **(Optional, Hermes users) Register the MCP server.**
   ```sh
   .venv/bin/curriculum mcp-register
   ```
   Runs `hermes mcp add curriculum ...` when `hermes` is on PATH; otherwise prints the
   exact command (with the key shown as `"$CURRICULUM_API_KEY"`, never echoed) to run where
   Hermes lives. Hermes then launches the server via `curriculum serve`, which `execv`s
   `python -m curriculum.mcp.server` and speaks MCP over stdio.

## How it works

**The OKF / Postgres split (polyglot, single-ownership).** Two stores, neither
authoritative for the same fact, so nothing drifts:
- The **OKF markdown bundle** (under `CURRICULUM_OKF_PATH`, default `./bundle`) owns the
  *content*: concept and question prose plus citations, as human-readable, git-diffable
  markdown files with YAML frontmatter (`docs/okf-spec.md`).
- **Postgres + pgvector** owns the *structure, metadata, and state*: concepts, edges and
  their importance, questions index, embeddings, and all per-learner FSRS/review state.

**The engine.** Pure domain + ports, so it runs and tests with in-memory + fake adapters
(no Postgres, no inference):
- **FSRS** -- the spaced-repetition scheduler (when each concept is next due).
- **FIRe** -- importance propagation across the edge graph after a graded answer.
- **Selection** -- a weighted-sampling policy over five scoring terms (urgency, difficulty
  fit, exploration, interleave penalty, coverage) that picks the single best next action.

**The build pipeline** (`curriculum.app.build`, driven by the CLI): `ingest` runs each
source through a graph-only multipass pipeline (extract -> dedupe -> infer-edges -> verify),
`link` connects still-isolated concepts via embedding-guided nearest-neighbour edge repair,
and `questions` generates batched single-concept and multi-hop exam questions over the
persisted graph.

**The MCP tools** (served by `curriculum serve`, a thin adapter over the
`CurriculumService` use-cases):
- `next(course)` -- the single best next action plus the ranked candidate field.
- `explain(concept_id)` -- grounded teaching prose for a concept.
- `quiz(concept_id, difficulty?)` -- a question (metadata + prompt/rubric) for a concept.
- `grade(concept_id, score, ...)` -- record a graded answer: update the schedule, run FIRe
  propagation, update skip counts, log calibration; returns the new schedule.
- `state(course)` -- a burndown / progress snapshot for the course.

## Reset

To start a course over (wipe the graph + all learner state and the generated content):
```sh
docker compose exec -T db psql -U curriculum -d curriculum \
  -c 'TRUNCATE concept, edge, question, learner_state, review_log, course_profile CASCADE;'
rm -rf bundle      # or whatever CURRICULUM_OKF_PATH points at
```
`make reset` does both. The database container and its volume stay up; only the rows and
the OKF bundle are removed, so the next `curriculum build` starts clean.

## Troubleshooting

- **`database ... unreachable` / connection refused** -- the DB is not up or not on :5433.
  Run `docker compose up -d db`, wait for the healthcheck, and re-check with
  `curriculum doctor`. If your Postgres is elsewhere, set `CURRICULUM_DB_URL`.
- **`CURRICULUM_API_KEY is not set`** -- every inference-backed stage (ingest/link/questions) needs an OpenAI-compatible provider key.
  `export CURRICULUM_API_KEY=...` in the same shell; legacy `NOUS_API_KEY` is still accepted. `doctor` reports it as MISS.
- **Embedding dimension mismatch** -- the schema declares `embedding vector(3072)`. If you
  switch `CURRICULUM_EMBED_MODEL` to a model of a different dimension (or set a different
  `CURRICULUM_EMBED_DIM`), edit the `vector(N)` in `schema/001_init.sql` to match and
  recreate the table, or persistence will reject the vectors.
- **`docker not found on PATH`** -- the `db-up` / `db-down` commands and `make reset` need
  the docker CLI. Install Docker, or start an external Postgres and point `CURRICULUM_DB_URL`
  at it.
- **`the 'mcp' package is not installed` / `adapter unavailable` for psycopg** -- you skipped
  an extra. Reinstall with `uv pip install -e '.[postgres,mcp,nous]'`.
- **`curriculum: docker-compose.yml not found`** -- run the `db-*` commands from the repo
  checkout (the compose file is a repo artifact, located by searching upward).
- **Partial ingest** (`files` < your source count) -- one or more sources failed (bad path,
  malformed text, a transient inference error); the batch tolerates it rather than discarding
  the work already committed. Check the failing source and re-run `curriculum ingest` (writes
  are idempotent/upsert).
- **`mcp-register` prints instead of running** -- `hermes` is not on this machine's PATH;
  copy the printed command and run it where Hermes is installed.

## Quick command reference

| Command                         | What it does |
|---------------------------------|--------------|
| `curriculum doctor`             | Check prerequisites (docker, DB, key, bundle); non-zero if any miss. |
| `curriculum db-up` / `db-down`  | Start / stop the Postgres+pgvector container. |
| `curriculum ingest <manifest>`  | Ingest a manifest's sources into the concept/edge graph. |
| `curriculum link [--course C]`  | Link isolated concepts via embedding-guided edge repair. |
| `curriculum questions [--course C]` | Generate exam questions over the persisted graph. |
| `curriculum build <manifest>`   | Full pipeline: ingest -> link -> questions (course from the manifest). |
| `curriculum status [--course C]`| Read-only graph counts for a course. |
| `curriculum serve`              | Become the stdio MCP server (Hermes launches this). |
| `curriculum mcp-register`       | Register the MCP server with Hermes (or print the command). |

Tests need neither the database nor a key: `make test` (or
`PYTHONPATH=src python3 -m unittest discover -s tests -t .`).
