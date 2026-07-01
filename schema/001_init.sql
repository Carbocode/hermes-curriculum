-- Curriculum layer schema (Postgres + pgvector).
-- Postgres owns: graph structure, metadata, embeddings, and per-learner state.
-- OKF owns: concept/question prose. The `concept` row holds a pointer
-- (content_hash) into the OKF bundle, never the body itself.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS concept (
    id            text PRIMARY KEY,                  -- OKF concept-id (bundle path without .md)
    course        text NOT NULL,
    title         text NOT NULL,
    description   text NOT NULL DEFAULT '',
    importance    real NOT NULL DEFAULT 0.5,
    source_refs   jsonb NOT NULL DEFAULT '[]',
    content_hash  text,                              -- sha256 of OKF content file (sync/staleness)
    embedding     vector(4096),                      -- derived from content; refreshed on hash change (nvidia/nv-embed-v1)
    status        text NOT NULL DEFAULT 'active',
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS concept_course_idx ON concept (course);

CREATE TABLE IF NOT EXISTS edge (
    src            text NOT NULL REFERENCES concept (id) ON DELETE CASCADE,
    dst            text NOT NULL REFERENCES concept (id) ON DELETE CASCADE,
    type           text NOT NULL CHECK (type IN ('prerequisite', 'encompasses', 'related')),
    weight         real NOT NULL DEFAULT 1.0,        -- encompasses: fraction of dst exercised, 0..1
    importance     real NOT NULL DEFAULT 0.5,
    rationale      text,
    source_ref     jsonb,
    exposure_count int NOT NULL DEFAULT 0,
    skip_count     int NOT NULL DEFAULT 0,
    last_traversed timestamptz,
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS edge_src_idx ON edge (src, type);
CREATE INDEX IF NOT EXISTS edge_dst_idx ON edge (dst, type);

CREATE TABLE IF NOT EXISTS question (
    id           text PRIMARY KEY,
    concept_id   text NOT NULL REFERENCES concept (id) ON DELETE CASCADE,
    edge_id      text,                               -- "src::type::dst" for multi-hop/connection Qs
    kind         text NOT NULL DEFAULT 'open',
    difficulty   int NOT NULL DEFAULT 1,
    hop_count    int NOT NULL DEFAULT 1,
    source_refs  jsonb NOT NULL DEFAULT '[]',
    generated_by text
);
CREATE INDEX IF NOT EXISTS question_concept_idx ON question (concept_id);
CREATE INDEX IF NOT EXISTS question_edge_idx ON question (edge_id);

CREATE TABLE IF NOT EXISTS learner_state (
    concept_id   text PRIMARY KEY REFERENCES concept (id) ON DELETE CASCADE,
    stability    real,
    difficulty   real,
    last_review  timestamptz,
    due_at       timestamptz,
    reps         int NOT NULL DEFAULT 0,
    lapses       int NOT NULL DEFAULT 0,
    mastery      text NOT NULL DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS learner_due_idx ON learner_state (due_at);

CREATE TABLE IF NOT EXISTS review_log (
    id            bigserial PRIMARY KEY,
    concept_id    text NOT NULL REFERENCES concept (id) ON DELETE CASCADE,
    question_id   text,
    grade         int,
    fsrs_rating   int,
    predicted     int,
    at            timestamptz NOT NULL DEFAULT now(),
    scheduler_ver text
);
CREATE INDEX IF NOT EXISTS review_concept_idx ON review_log (concept_id);

CREATE TABLE IF NOT EXISTS course_profile (
    course            text PRIMARY KEY,
    archetype         text NOT NULL,
    exam_format       jsonb NOT NULL DEFAULT '{}',
    weights           jsonb NOT NULL DEFAULT '{}',
    target_retention  real NOT NULL DEFAULT 0.90,
    exam_date         date,
    confirmed_by_user boolean NOT NULL DEFAULT false
);
