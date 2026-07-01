"""Tests for runtime settings environment loading."""
from __future__ import annotations

import unittest

from curriculum.config import load


class ProviderSettingsTest(unittest.TestCase):
    def test_generic_openai_compatible_env_wins(self) -> None:
        settings = load(
            {
                "CURRICULUM_API_KEY": "generic-key",
                "NOUS_API_KEY": "legacy-key",
                "CURRICULUM_BASE_URL": "https://example.test/v1",
                "NOUS_BASE_URL": "https://legacy.test/v1",
                "CURRICULUM_INGEST_MODEL": "vendor/chat-model",
                "CURRICULUM_EMBED_MODEL": "vendor/embed-model",
                "CURRICULUM_EMBED_DIM": "1024",
            }
        )

        self.assertEqual(settings.api_key, "generic-key")
        self.assertEqual(settings.base_url, "https://example.test/v1")
        self.assertEqual(settings.ingest_model, "vendor/chat-model")
        self.assertEqual(settings.embed_model, "vendor/embed-model")
        self.assertEqual(settings.embedding_dim, 1024)

    def test_legacy_nous_env_still_works_as_fallback(self) -> None:
        settings = load(
            {
                "NOUS_API_KEY": "legacy-key",
                "NOUS_BASE_URL": "https://legacy.test/v1",
            }
        )

        self.assertEqual(settings.api_key, "legacy-key")
        self.assertEqual(settings.base_url, "https://legacy.test/v1")
        self.assertEqual(settings.nous_api_key, "legacy-key")
        self.assertEqual(settings.nous_base_url, "https://legacy.test/v1")

    def test_defaults_remain_nous_compatible(self) -> None:
        settings = load({})

        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.base_url, "https://inference-api.nousresearch.com/v1")
        self.assertEqual(settings.ingest_model, "deepseek/deepseek-v4-flash")
        self.assertEqual(settings.embed_model, "google/gemini-embedding-2")
        self.assertEqual(settings.embedding_dim, 3072)


if __name__ == "__main__":
    unittest.main()
