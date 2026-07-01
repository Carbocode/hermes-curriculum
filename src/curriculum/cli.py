"""Agent-facing command line: the single entrypoint that makes the whole engine
reproducible and bootstrappable from a fresh checkout.

This is the outermost driver of the hexagon's *build* side (the MCP server is the
outermost driver of its *serve* side). It owns no business logic: every command
is a thin shell over :mod:`curriculum.app.build`, the docker-compose database, or
the MCP server. Its job is purely to turn argv + the environment-driven
:class:`curriculum.config.Settings` into one of those calls and a process exit
code, so a human or an agent can stand the system up with ``curriculum build
corpus.json`` instead of a pile of ad-hoc scripts.

Why every heavy import is deferred into the handlers
----------------------------------------------------
``curriculum --help`` and ``curriculum doctor`` MUST work on a machine that has
neither ``psycopg`` nor the ``mcp`` SDK installed -- that is the first thing an
operator runs to find out *what is missing*. So this module imports only the
standard library plus the light, dependency-free ``config``/``errors`` modules at
load time; :mod:`curriculum.app.build` (which can pull in the Postgres adapter)
and the MCP server are imported INSIDE the command handlers that need them. A
fresh-checkout ``doctor`` therefore diagnoses the very dependencies the other
commands require, rather than crashing on an ImportError before it can report.

Exit codes are returned, never raised: ``main`` converts argparse's ``SystemExit``
(``--help`` -> 0, a usage error -> 2) and any adapter/driver failure into an
integer, so callers, tests, and shell scripts get a clean status without having
to catch exceptions.

Standard library only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .config import load as load_settings
from .domain.errors import CurriculumError

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# Small shared helpers.
# --------------------------------------------------------------------------- #
def _emit(obj: object) -> None:
    """Print a result as stable, indented JSON.

    ``sort_keys`` makes the output deterministic so an agent (or a test) can diff
    successive runs, and JSON keeps the machine-readable contract uniform across
    every data-returning command.
    """
    print(json.dumps(obj, indent=2, sort_keys=True))


def _course(args: argparse.Namespace, settings: Settings) -> str:
    """Resolve the course: an explicit ``--course`` wins, else the configured
    default, so single-course setups never have to pass the flag."""
    return args.course or settings.default_course


def _repo_root() -> Path | None:
    """Locate the directory that owns ``docker-compose.yml`` (the repo root).

    The compose file is a repo artifact, not something installed alongside the
    package, so we search upward from the working directory first (the operator
    usually runs ``curriculum db-up`` from the checkout) and then from this
    module's own location (covers an editable install invoked from elsewhere).
    ``None`` means no compose file was found -- the db commands then fail with an
    actionable message instead of running compose in the wrong place.
    """
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / "docker-compose.yml").is_file():
                return directory
    return None


def _compose(compose_args: list[str]) -> int:
    """Run ``docker compose <compose_args>`` in the repo root, returning its code.

    Guards both preconditions up front -- the docker binary and the compose file
    -- because a missing-binary ``FileNotFoundError`` or a wrong-directory compose
    run are exactly the confusing failures this wrapper exists to prevent.
    """
    if shutil.which("docker") is None:
        print("curriculum: docker not found on PATH", file=sys.stderr)
        return 1
    root = _repo_root()
    if root is None:
        print(
            "curriculum: docker-compose.yml not found "
            "(run from the repository checkout)",
            file=sys.stderr,
        )
        return 1
    result = subprocess.run(["docker", "compose", *compose_args], cwd=str(root))
    return result.returncode


class _Tee:
    """Mirror writes to every wrapped stream immediately."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)  # type: ignore[attr-defined]
            stream.flush()  # type: ignore[attr-defined]
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()  # type: ignore[attr-defined]


def _command_log_path(command: str) -> Path:
    """Return the predictable log file for a build-side command."""
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return logs_dir / f"curriculum-{command}-{stamp}-pid{os.getpid()}.log"


@contextmanager
def _tee_command_output(command: str):
    """Mirror stdout/stderr to a durable log file while a command runs."""
    log_path = _command_log_path(command)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"# curriculum {command}\n")
        log.write(f"# pid={os.getpid()}\n")
        log.write(f"# started={datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        try:
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError):
            stdout, stderr = sys.stdout, sys.stderr
            with redirect_stdout(_Tee(stdout, log)), redirect_stderr(_Tee(stderr, log)):
                print(f"[curriculum] logging to {log_path}")
                yield log_path
            log.write(f"# finished={datetime.now(timezone.utc).isoformat()}\n")
            log.flush()
            return

        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        os.dup2(out_w, stdout_fd)
        os.dup2(err_w, stderr_fd)
        os.close(out_w)
        os.close(err_w)

        def _pump(read_fd: int) -> None:
            with os.fdopen(read_fd, "rb", closefd=True) as reader:
                while True:
                    chunk = reader.read(8192)
                    if not chunk:
                        break
                    log.write(chunk.decode("utf-8", errors="replace"))
                    log.flush()

        threads = [
            threading.Thread(target=_pump, args=(out_r,), daemon=True),
            threading.Thread(target=_pump, args=(err_r,), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            print(f"[curriculum] logging to {log_path}")
            yield log_path
        finally:
            try:
                sys.stdout.flush()
            finally:
                sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
            for thread in threads:
                thread.join(timeout=1)
            log.write(f"# finished={datetime.now(timezone.utc).isoformat()}\n")
            log.flush()


# --------------------------------------------------------------------------- #
# Build-side command handlers (each lazy-imports curriculum.app.build).
# --------------------------------------------------------------------------- #
def _cmd_db_up(args: argparse.Namespace, settings: Settings) -> int:
    """Start the bundled Postgres+pgvector service in the background."""
    return _compose(["up", "-d", "db"])


def _cmd_db_down(args: argparse.Namespace, settings: Settings) -> int:
    """Stop the compose stack (leaves the named data volume intact)."""
    return _compose(["down"])


def _cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    """Print the read-only graph counts for a course (no inference, no writes)."""
    from .app import build

    _emit(build.status(settings, _course(args, settings)))
    return 0


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest a manifest's sources into the concept/edge graph."""
    from .app import build

    manifest = build.load_manifest(args.manifest)
    with _tee_command_output("ingest"):
        _emit(build.ingest(manifest, settings))
    return 0


def _cmd_link(args: argparse.Namespace, settings: Settings) -> int:
    """Link isolated concepts via embedding-guided edge repair."""
    from .app import build

    with _tee_command_output("link"):
        _emit(build.link(settings, _course(args, settings)))
    return 0


def _cmd_questions(args: argparse.Namespace, settings: Settings) -> int:
    """Generate exam questions over the persisted graph (batched)."""
    from .app import build

    with _tee_command_output("questions"):
        _emit(build.generate_questions(settings, _course(args, settings)))
    return 0


def _cmd_build(args: argparse.Namespace, settings: Settings) -> int:
    """Run the full pipeline for a manifest: ingest -> link -> questions.

    The course for the link/question stages comes from the manifest itself (not
    ``--course``): a build is scoped to exactly the course it ingests. Each
    stage's result is emitted as it completes so a long build shows progress
    rather than going dark until the end.
    """
    from .app import build

    manifest = build.load_manifest(args.manifest)
    course = manifest["course"]
    with _tee_command_output("build"):
        _emit({"stage": "ingest", "result": build.ingest(manifest, settings)})
        _emit({"stage": "link", "result": build.link(settings, course)})
        _emit(
            {
                "stage": "questions",
                "result": build.generate_questions(settings, course),
            }
        )
    return 0


# --------------------------------------------------------------------------- #
# Serve-side and registration handlers.
# --------------------------------------------------------------------------- #
def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Become the stdio MCP server (this is what Hermes launches).

    ``os.execv`` REPLACES this process with ``python -m curriculum.mcp.server``,
    so Hermes talks to the server directly over the stdio it opened for
    ``curriculum serve`` -- no wrapper process sits between them holding the pipes.
    Nothing is printed (stdout is the MCP transport). On success execv does not
    return; the trailing ``return`` only covers the impossible fall-through, and
    an ``OSError`` propagates to ``main`` to become a non-zero exit.
    """
    os.execv(sys.executable, [sys.executable, "-m", "curriculum.mcp.server"])
    return 0  # pragma: no cover - unreachable: execv never returns on success


def _register_argv(python: str, settings: Settings, key_value: str) -> list[str]:
    """Build the ``hermes mcp add`` argv that registers this MCP server.

    The bundle path is absolutised because Hermes spawns the server from its own
    (arbitrary) working directory, so a relative ``./bundle`` would not resolve.
    ``key_value`` is injected by the caller so the runnable form can carry the
    real key while the printed form carries a shell reference (see
    :func:`_cmd_mcp_register`).
    """
    bundle = os.path.abspath(settings.okf_bundle_path)
    return [
        "hermes", "mcp", "add", "curriculum",
        "--command", python,
        "--env",
        f"CURRICULUM_API_KEY={key_value}",
        f"CURRICULUM_BASE_URL={settings.base_url}",
        f"CURRICULUM_DB_URL={settings.database_url}",
        f"CURRICULUM_OKF_PATH={bundle}",
        f"CURRICULUM_INGEST_MODEL={settings.ingest_model}",
        f"CURRICULUM_EMBED_MODEL={settings.embed_model}",
        f"CURRICULUM_EMBED_DIM={settings.embedding_dim}",
        "--args", "-m", "curriculum.mcp.server",
    ]


def _render(argv: list[str]) -> str:
    """Render an argv as a copy-pasteable shell line, keeping the key a reference.

    Every token is shell-quoted so paths with spaces survive, EXCEPT the
    ``CURRICULUM_API_KEY`` assignment, which is emitted as
    ``CURRICULUM_API_KEY="$CURRICULUM_API_KEY"`` so the secret never lands in
    paste time) while the line stays runnable.
    """
    parts: list[str] = []
    for token in argv:
        if token.startswith("CURRICULUM_API_KEY="):
            parts.append('CURRICULUM_API_KEY="$CURRICULUM_API_KEY"')
        else:
            parts.append(shlex.quote(token))
    return " ".join(parts)


def _cmd_mcp_register(args: argparse.Namespace, settings: Settings) -> int:
    """Register this MCP server with Hermes, or print the command if hermes is absent.

    Convenience wiring for the one supported host (Hermes). When ``hermes`` is on
    PATH we run the registration with the resolved key so it actually lands; when
    it is not, we print the exact command (key shown as a shell reference) for the
    operator to run wherever Hermes lives. The printed form is shown in both cases
    so the operator can see what ran without the secret being echoed.
    """
    python = sys.executable
    printable = _render(_register_argv(python, settings, ""))
    if shutil.which("hermes") is None:
        print("hermes not found on PATH; run this where Hermes is installed:")
        print(printable)
        return 0
    print(f"running: {printable}")
    result = subprocess.run(
        _register_argv(python, settings, settings.api_key or "")
    )
    return result.returncode


# --------------------------------------------------------------------------- #
# doctor: a fresh-machine readiness checklist (each probe is isolated + safe).
# --------------------------------------------------------------------------- #
def _check_docker() -> tuple[str, bool, str]:
    """Is the docker binary available (needed for ``db-up``)?"""
    path = shutil.which("docker")
    if path:
        return ("docker", True, path)
    return ("docker", False, "not found on PATH")


def _check_db(settings: Settings) -> tuple[str, bool, str]:
    """Can we open the configured database and run a trivial query?

    The Postgres adapter is imported lazily and EVERY failure mode is folded into
    a reported line -- driver missing (``connect`` raises a RuntimeError), refused
    connection, or a failing probe query -- so ``doctor`` never raises; it only
    reports. The connection is always closed.
    """
    try:
        from .storage.postgres import connect
    except ImportError as exc:  # pragma: no cover - module imports without driver
        return ("database", False, f"adapter unavailable: {exc}")
    try:
        connection = connect(settings.database_url)
    except Exception as exc:  # noqa: BLE001 - driver-missing or refused: report, not raise
        return ("database", False, f"unreachable: {exc}")
    try:
        connection.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - connected but unhealthy
        return ("database", False, f"query failed: {exc}")
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - close errors are not actionable here
            pass
    return ("database", True, settings.database_url)


def _check_api_key(settings: Settings) -> tuple[str, bool, str]:
    """Is an OpenAI-compatible provider API key set?"""
    if settings.api_key:
        return ("CURRICULUM_API_KEY", True, "set")
    return (
        "CURRICULUM_API_KEY",
        False,
        "not set (export CURRICULUM_API_KEY; legacy NOUS_API_KEY also works)",
    )


def _check_bundle(settings: Settings) -> tuple[str, bool, str]:
    """Does the OKF content bundle directory exist yet?

    Reported as missing before the first build (the ingest step creates it), so a
    pre-build ``doctor`` honestly shows it is not there rather than implying the
    corpus is ready.
    """
    path = os.path.abspath(settings.okf_bundle_path)
    if os.path.isdir(path):
        return ("okf bundle", True, path)
    return ("okf bundle", False, f"{path} (missing; created on first ingest)")


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Print an OK/MISS checklist of the build prerequisites; non-zero if any miss.

    Returns a non-zero code when something is missing so the command doubles as a
    scriptable readiness gate, while still printing the full picture either way.
    """
    checks = [
        _check_docker(),
        _check_db(settings),
        _check_api_key(settings),
        _check_bundle(settings),
    ]
    all_ok = True
    for label, ok, detail in checks:
        mark = " OK " if ok else "MISS"
        print(f"[{mark}] {label}: {detail}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# Parser wiring + entrypoint.
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser and bind each subcommand to its handler.

    Subcommands carry their handler via ``set_defaults(func=...)`` so ``main``
    dispatches uniformly; the top parser defaults ``func`` to ``None`` so a
    no-subcommand invocation is detectable (and answered with the help text)
    rather than crashing on a missing attribute.
    """
    parser = argparse.ArgumentParser(
        prog="curriculum",
        description=(
            "Build and serve the curriculum knowledge-graph engine: stand up the "
            "database, ingest a corpus manifest into the graph, link and generate "
            "questions, and serve the result to Hermes over MCP."
        ),
    )
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # A shared --course option for the course-scoped read/build steps.
    course_parent = argparse.ArgumentParser(add_help=False)
    course_parent.add_argument(
        "--course",
        default=None,
        help="course name (defaults to settings.default_course)",
    )

    p_db_up = sub.add_parser("db-up", help="start the Postgres+pgvector container")
    p_db_up.set_defaults(func=_cmd_db_up)

    p_db_down = sub.add_parser("db-down", help="stop the compose stack")
    p_db_down.set_defaults(func=_cmd_db_down)

    p_status = sub.add_parser(
        "status", parents=[course_parent], help="print graph counts for a course"
    )
    p_status.set_defaults(func=_cmd_status)

    p_ingest = sub.add_parser("ingest", help="ingest a manifest into the graph")
    p_ingest.add_argument("manifest", help="path to the corpus manifest JSON")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_link = sub.add_parser(
        "link", parents=[course_parent], help="link isolated concepts"
    )
    p_link.set_defaults(func=_cmd_link)

    p_questions = sub.add_parser(
        "questions", parents=[course_parent], help="generate exam questions"
    )
    p_questions.set_defaults(func=_cmd_questions)

    p_build = sub.add_parser(
        "build", help="full pipeline: ingest -> link -> questions"
    )
    p_build.add_argument("manifest", help="path to the corpus manifest JSON")
    p_build.set_defaults(func=_cmd_build)

    p_serve = sub.add_parser("serve", help="run the stdio MCP server (for Hermes)")
    p_serve.set_defaults(func=_cmd_serve)

    p_register = sub.add_parser(
        "mcp-register", help="register this MCP server with Hermes"
    )
    p_register.set_defaults(func=_cmd_mcp_register)

    p_doctor = sub.add_parser(
        "doctor", help="check prerequisites (docker, DB, key, bundle)"
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen command, returning a process exit code.

    Argparse's ``SystemExit`` is caught and converted to its integer code so that
    ``--help`` (0) and usage errors (2) return cleanly instead of raising. With no
    subcommand we print the help text and return 0. All command/adapter failures
    are funnelled into a single non-zero exit with a concise stderr message:
    :class:`CurriculumError` for domain problems (bad manifest, missing key) and a
    broad catch for driver/DB/subprocess failures, since the CLI is a boundary and
    a clean status beats a stack trace.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0

    try:
        settings = load_settings()
        return args.func(args, settings)
    except CurriculumError as exc:
        print(f"curriculum: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - boundary: driver/DB/subprocess -> clean exit
        print(f"curriculum: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
