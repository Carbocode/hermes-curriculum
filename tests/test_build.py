"""Tests for the config-driven build orchestration (curriculum.app.build).

Stdlib unittest only and deliberately offline. The manifest loader is pure
(JSON parse + validation), so it is unit-tested directly against throwaway temp
files -- happy path, the ``chunk_lines`` default, and every malformed shape that
must raise :class:`ConfigError`. The network/DB orchestration functions
(``ingest``/``link``/``generate_questions``/``status``) are NOT exercised here:
their collaborators -- the ingestion pipeline and the embedding linker -- own
those tests. What we DO assert is the load-bearing property that makes this
module safe to import everywhere: it imports with neither ``psycopg`` nor a
network present (the heavy collaborators are deferred to call time).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from curriculum.app import build
from curriculum.config import Settings
from curriculum.domain.errors import ConfigError


def _write_manifest(directory: str, payload: object) -> str:
    """Write ``payload`` as JSON into ``directory`` and return its path."""
    path = Path(directory) / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class LoadManifestHappyPathTest(unittest.TestCase):
    def test_full_manifest_is_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {
                    "course": "Cybersecurity",
                    "chunk_lines": 80,
                    "sources": [
                        {"path": "materials/a.txt", "token": "a"},
                        # Extra keys on a source are dropped by normalisation.
                        {"path": "materials/b.txt", "token": "b", "note": "ignored"},
                    ],
                },
            )
            manifest = build.load_manifest(path)

        self.assertEqual(manifest["course"], "Cybersecurity")
        self.assertEqual(manifest["chunk_lines"], 80)
        self.assertEqual(
            manifest["sources"],
            [
                {"path": "materials/a.txt", "token": "a"},
                {"path": "materials/b.txt", "token": "b"},
            ],
        )

    def test_chunk_lines_defaults_to_150_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {"course": "C", "sources": [{"path": "a.txt", "token": "a"}]},
            )
            manifest = build.load_manifest(path)
        self.assertEqual(manifest["chunk_lines"], 150)

    def test_example_manifest_shape_loads(self) -> None:
        # The shipped corpus.example.json shape (minus its leading _comment) must
        # be accepted, since it is the documented template users copy.
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                directory,
                {
                    "_comment": "irrelevant top-level extra",
                    "course": "YourCourse",
                    "chunk_lines": 150,
                    "sources": [
                        {"path": "materials/lecture-01-intro.txt", "token": "lecture-01-intro"}
                    ],
                },
            )
            manifest = build.load_manifest(path)
        self.assertEqual(manifest["course"], "YourCourse")
        self.assertEqual(len(manifest["sources"]), 1)


class LoadManifestMalformedTest(unittest.TestCase):
    def _assert_rejects(self, payload: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(directory, payload)
            with self.assertRaises(ConfigError):
                build.load_manifest(path)

    def test_top_level_not_an_object(self) -> None:
        self._assert_rejects([{"path": "a", "token": "a"}])

    def test_missing_course(self) -> None:
        self._assert_rejects({"sources": [{"path": "a", "token": "a"}]})

    def test_blank_course(self) -> None:
        self._assert_rejects({"course": "   ", "sources": [{"path": "a", "token": "a"}]})

    def test_non_string_course(self) -> None:
        self._assert_rejects({"course": 1, "sources": [{"path": "a", "token": "a"}]})

    def test_missing_sources(self) -> None:
        self._assert_rejects({"course": "C"})

    def test_empty_sources(self) -> None:
        self._assert_rejects({"course": "C", "sources": []})

    def test_sources_not_a_list(self) -> None:
        self._assert_rejects({"course": "C", "sources": {"path": "a", "token": "a"}})

    def test_source_not_an_object(self) -> None:
        self._assert_rejects({"course": "C", "sources": ["a.txt"]})

    def test_source_missing_path(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"token": "a"}]})

    def test_source_missing_token(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"path": "a.txt"}]})

    def test_source_blank_token(self) -> None:
        self._assert_rejects({"course": "C", "sources": [{"path": "a.txt", "token": ""}]})

    def test_non_positive_chunk_lines(self) -> None:
        self._assert_rejects(
            {"course": "C", "sources": [{"path": "a", "token": "a"}], "chunk_lines": 0}
        )

    def test_boolean_chunk_lines_rejected(self) -> None:
        # bool is an int subclass; it must not slip through as a "valid integer".
        self._assert_rejects(
            {"course": "C", "sources": [{"path": "a", "token": "a"}], "chunk_lines": True}
        )

    def test_invalid_json_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text("{ not valid json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                build.load_manifest(str(path))

    def test_missing_file_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "does-not-exist.json")
            with self.assertRaises(ConfigError):
                build.load_manifest(missing)


class ModuleImportsOfflineTest(unittest.TestCase):
    """The module must expose its public API without importing any heavy driver."""

    def test_public_functions_are_callable(self) -> None:
        for name in ("load_manifest", "ingest", "link", "generate_questions", "status"):
            self.assertTrue(callable(getattr(build, name)), name)

    def test_package_reexports_match(self) -> None:
        import curriculum.app as app

        for name in app.__all__:
            self.assertIs(getattr(app, name), getattr(build, name))


class ProviderRequirementTest(unittest.TestCase):
    def test_require_api_key_accepts_generic_key(self) -> None:
        build._require_api_key(Settings(api_key="key"))

    def test_require_api_key_mentions_generic_and_legacy_names(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            build._require_api_key(Settings(api_key=None))
        message = str(ctx.exception)
        self.assertIn("CURRICULUM_API_KEY", message)
        self.assertIn("NOUS_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
