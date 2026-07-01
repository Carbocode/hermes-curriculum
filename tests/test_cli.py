"""Tests for the agent-facing CLI (curriculum.cli).

Stdlib unittest only and strictly offline: the contract these tests pin is the
*shell* of the CLI -- argv parsing, exit codes, and that the dependency-free
commands (``--help`` and ``doctor``) work on a fresh machine -- NOT the build
orchestration, whose collaborators own those tests. So we never call a command
that would touch the network or the database: the only data-driven command we
exercise is ``doctor``, with its database probe monkeypatched to a canned result
so no socket is ever opened.

The load-bearing properties asserted here:

* ``main`` returns an ``int`` and never raises, even on argparse's own exit paths
  (``--help`` -> 0, an unknown subcommand -> non-zero), because it converts
  ``SystemExit`` into a return code.
* a bare invocation prints usage instead of crashing.
* ``doctor`` runs end to end (every probe isolated) and reports a checklist.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from curriculum import cli
from curriculum.config import Settings


class HelpAndUsageTest(unittest.TestCase):
    def test_help_returns_int_and_prints_usage(self) -> None:
        # --help makes argparse print to stdout and raise SystemExit(0); main must
        # absorb that and hand back a plain int.
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--help"])
        self.assertIsInstance(code, int)
        self.assertEqual(code, 0)
        self.assertIn("usage", out.getvalue().lower())

    def test_no_args_prints_usage_and_returns_int(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main([])
        self.assertIsInstance(code, int)
        self.assertIn("usage", out.getvalue().lower())

    def test_unknown_subcommand_returns_nonzero(self) -> None:
        # An invalid choice makes argparse print an error to stderr and exit 2;
        # the caller should see a non-zero int, not an exception.
        err = io.StringIO()
        with redirect_stderr(err):
            code = cli.main(["definitely-not-a-command"])
        self.assertIsInstance(code, int)
        self.assertNotEqual(code, 0)


class DoctorTest(unittest.TestCase):
    def test_doctor_runs_without_raising(self) -> None:
        # Pin the DB probe so the test opens no socket (honouring the no-network /
        # no-DB rule) while the docker/key/bundle probes run for real -- they only
        # consult PATH, settings, and the filesystem, so they are side-effect free.
        out = io.StringIO()
        with mock.patch.object(
            cli, "_check_db", return_value=("database", False, "skipped in test")
        ):
            with redirect_stdout(out):
                code = cli.main(["doctor"])
        self.assertIsInstance(code, int)
        printed = out.getvalue().lower()
        # Every probe is reported, each line marked OK or MISS.
        self.assertIn("docker", printed)
        self.assertIn("database", printed)
        self.assertIn("curriculum_api_key", printed)
        self.assertTrue("ok" in printed or "miss" in printed)

    def test_doctor_returns_nonzero_when_a_check_misses(self) -> None:
        # With the DB probe forced to miss, doctor must report a non-zero status
        # so it can double as a scriptable readiness gate.
        with mock.patch.object(
            cli, "_check_db", return_value=("database", False, "skipped in test")
        ):
            with redirect_stdout(io.StringIO()):
                code = cli.main(["doctor"])
        self.assertNotEqual(code, 0)


class McpRegisterCommandTest(unittest.TestCase):
    def test_register_command_uses_generic_provider_env_names(self) -> None:
        settings = Settings(api_key="secret", base_url="https://vendor.test/v1")
        argv = cli._register_argv("/python", settings, "secret")

        self.assertIn("CURRICULUM_API_KEY=secret", argv)
        self.assertIn("CURRICULUM_BASE_URL=https://vendor.test/v1", argv)
        self.assertIn("CURRICULUM_INGEST_MODEL=deepseek/deepseek-v4-flash", argv)
        self.assertIn("CURRICULUM_EMBED_MODEL=google/gemini-embedding-2", argv)
        self.assertIn("CURRICULUM_EMBED_DIM=3072", argv)
        self.assertNotIn("NOUS_API_KEY=secret", argv)

    def test_render_keeps_generic_key_as_shell_reference(self) -> None:
        rendered = cli._render(
            [
                "hermes",
                "mcp",
                "add",
                "curriculum",
                "--env",
                "CURRICULUM_API_KEY=",
            ]
        )

        self.assertIn('CURRICULUM_API_KEY="$CURRICULUM_API_KEY"', rendered)


class BuildLoggingTest(unittest.TestCase):
    def test_build_creates_a_durable_log_file(self) -> None:
        from curriculum.app import build as build_mod

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.json"
            corpus.write_text(
                json.dumps(
                    {
                        "course": "Algoritmi e Strutture dati",
                        "sources": [{"path": "materials/a.txt", "token": "a"}],
                    }
                ),
                encoding="utf-8",
            )
            previous = os.getcwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(
                        build_mod,
                        "load_manifest",
                        return_value={"course": "Algoritmi e Strutture dati"},
                    ),
                    mock.patch.object(build_mod, "ingest", return_value={"concepts": 1}),
                    mock.patch.object(build_mod, "link", return_value={"edges": 2}),
                    mock.patch.object(
                        build_mod, "generate_questions", return_value={"questions": 3}
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    code = cli.main(["build", str(corpus)])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            logs = sorted((root / "logs").glob("curriculum-build-*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertIn('"stage": "ingest"', text)
            self.assertIn('"stage": "questions"', text)
            self.assertIn("[curriculum] logging to", text)


if __name__ == "__main__":
    unittest.main()
