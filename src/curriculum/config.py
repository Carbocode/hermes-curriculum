"""Runtime settings, loaded from the environment.

Single source of configuration; adapters receive their settings explicitly
(no global lookups deep in the code). Defaults assume the bundled
docker-compose Postgres on port 5433.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str = "postgresql://curriculum:curriculum@localhost:5433/curriculum"
    okf_bundle_path: str = "./bundle"
    default_course: str = "Cybersecurity"
    api_key: str | None = None
    base_url: str = "https://inference-api.nousresearch.com/v1"
    ingest_model: str = "deepseek/deepseek-v4-flash"
    embed_model: str = "google/gemini-embedding-2"  # multimodal, native 3072-dim
    embedding_dim: int = 3072

    @property
    def nous_api_key(self) -> str | None:
        """Backward-compatible alias for the generic provider API key."""
        return self.api_key

    @property
    def nous_base_url(self) -> str:
        """Backward-compatible alias for the generic provider base URL."""
        return self.base_url


def load(env: dict[str, str] | None = None) -> Settings:
    e = os.environ if env is None else env
    d = Settings()  # instance access yields real defaults (class access yields slot descriptors)
    return Settings(
        database_url=e.get("CURRICULUM_DB_URL", d.database_url),
        okf_bundle_path=e.get("CURRICULUM_OKF_PATH", d.okf_bundle_path),
        default_course=e.get("CURRICULUM_COURSE", d.default_course),
        api_key=e.get("CURRICULUM_API_KEY", e.get("NOUS_API_KEY", d.api_key)),
        base_url=e.get("CURRICULUM_BASE_URL", e.get("NOUS_BASE_URL", d.base_url)),
        ingest_model=e.get("CURRICULUM_INGEST_MODEL", d.ingest_model),
        embed_model=e.get("CURRICULUM_EMBED_MODEL", d.embed_model),
        embedding_dim=int(e.get("CURRICULUM_EMBED_DIM", str(d.embedding_dim))),
    )
