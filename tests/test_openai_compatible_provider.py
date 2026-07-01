"""Tests for the generic OpenAI-compatible provider adapter."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from curriculum.providers_openai_compatible import (
    OpenAICompatibleEmbedder,
    OpenAICompatibleLlm,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OpenAICompatibleLlmTest(unittest.TestCase):
    def test_complete_posts_chat_completion_request(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):
            seen["url"] = req.full_url
            seen["headers"] = dict(req.header_items())
            seen["body"] = json.loads(req.data.decode("utf-8"))
            seen["timeout"] = timeout
            return _FakeResponse({"choices": [{"message": {"content": "hello"}}]})

        llm = OpenAICompatibleLlm(
            api_key="key",
            base_url="https://vendor.test/v1",
            model="vendor/chat",
            timeout=12,
        )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = llm.complete("Prompt", system="System", max_tokens=123, temperature=0.4)

        self.assertEqual(result, "hello")
        self.assertEqual(seen["url"], "https://vendor.test/v1/chat/completions")
        self.assertEqual(seen["headers"].get("Authorization"), "Bearer key")
        self.assertEqual(seen["body"]["model"], "vendor/chat")
        self.assertEqual(seen["body"]["max_tokens"], 123)
        self.assertEqual(seen["body"]["temperature"], 0.4)
        self.assertEqual(seen["body"]["messages"][0], {"role": "system", "content": "System"})
        self.assertEqual(seen["body"]["messages"][1], {"role": "user", "content": "Prompt"})
        self.assertEqual(seen["timeout"], 12)


class OpenAICompatibleEmbedderTest(unittest.TestCase):
    def test_embed_posts_embeddings_request_and_preserves_index_order(self) -> None:
        seen = {}

        def fake_urlopen(req, timeout):
            seen["url"] = req.full_url
            seen["headers"] = dict(req.header_items())
            seen["body"] = json.loads(req.data.decode("utf-8"))
            seen["timeout"] = timeout
            return _FakeResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0.2]},
                        {"index": 0, "embedding": [0.1]},
                    ]
                }
            )

        embedder = OpenAICompatibleEmbedder(
            api_key="key",
            base_url="https://vendor.test/v1/",
            model="vendor/embed",
            dim=1,
            timeout=7,
            batch=64,
        )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = embedder.embed(["a", "b"])

        self.assertEqual(result, [[0.1], [0.2]])
        self.assertEqual(seen["url"], "https://vendor.test/v1/embeddings")
        self.assertEqual(seen["headers"].get("Authorization"), "Bearer key")
        self.assertEqual(seen["body"], {"model": "vendor/embed", "input": ["a", "b"]})
        self.assertEqual(seen["timeout"], 7)


class NousCompatibilityAliasTest(unittest.TestCase):
    def test_legacy_nous_classes_remain_importable(self) -> None:
        from curriculum.providers_nous import NousEmbedder, NousLlm

        self.assertTrue(issubclass(NousLlm, OpenAICompatibleLlm))
        self.assertTrue(issubclass(NousEmbedder, OpenAICompatibleEmbedder))


if __name__ == "__main__":
    unittest.main()
