"""Workspace paths, artifact integrity validation, and process locking."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .environment import runtime_environment
from .io_utils import read_json, read_jsonl, sha256_json
from .lexical import LEXICAL_VERSION
from .manifest import ANALYSIS_VERSION, JUDGE_VERSION, ExperimentManifest
from .models import SMOKE_PROFILE_ID
from .protocol import CONDITIONS, PROTOCOL_HASH


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @property
    def directory(self) -> Path:
        return self.root

    @property
    def config_directory(self) -> Path:
        return self.root / "config"

    @property
    def data_directory(self) -> Path:
        return self.root / "data"

    @property
    def state_directory(self) -> Path:
        return self.root / "state"

    @property
    def results_directory(self) -> Path:
        return self.root / "results"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def questions(self) -> Path:
        return self.data_directory / "questions.jsonl"

    @property
    def models(self) -> Path:
        return self.config_directory / "models.json"

    @property
    def dataset_audit(self) -> Path:
        return self.data_directory / "dataset-audit.json"

    @property
    def environment(self) -> Path:
        return self.config_directory / "environment.json"

    @property
    def lexical_reference(self) -> Path:
        return self.data_directory / "lexical-reference.json"

    @property
    def database(self) -> Path:
        return self.state_directory / "run.sqlite3"

    @property
    def routes(self) -> Path:
        return self.config_directory / "routes.json"

    @property
    def preflight_log(self) -> Path:
        return self.config_directory / "preflight.json"

    @property
    def analysis(self) -> Path:
        return self.results_directory / "analysis.json"

    @property
    def report(self) -> Path:
        return self.results_directory / "report.txt"

    @property
    def cells_export(self) -> Path:
        return self.results_directory / "cells-and-judgments.jsonl"

    @property
    def attempts_export(self) -> Path:
        return self.results_directory / "attempts.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.root / ".lock"

    def create(self) -> None:
        for path in (
            self.root,
            self.config_directory,
            self.data_directory,
            self.state_directory,
            self.results_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> ExperimentManifest:
        value = read_json(self.manifest)
        if not isinstance(value, dict):
            raise ValueError("manifest must be a JSON object")
        return ExperimentManifest.from_dict(value)

    def require_prepared(self, *, require_runtime_environment: bool = True) -> ExperimentManifest:
        required = (
            self.manifest,
            self.questions,
            self.models,
            self.dataset_audit,
            self.environment,
            self.lexical_reference,
            self.database,
        )
        missing = [str(path.relative_to(self.root)) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("workspace is not prepared; missing " + ", ".join(missing))
        manifest = self.load_manifest()
        questions = read_jsonl(self.questions)
        models = read_json(self.models)
        environment = read_json(self.environment)
        if not isinstance(models, list):
            raise ValueError("models.json must contain a list")
        environment_matches_manifest = (
            isinstance(environment, dict) and sha256_json(environment) == manifest.environment_hash
        )
        checks = {
            "package_version": manifest.package_version == __version__,
            "questions_hash": sha256_json(questions) == manifest.questions_hash,
            "question_ids": tuple(str(row.get("id")) for row in questions) == manifest.question_ids,
            "models_hash": sha256_json(models) == manifest.models_hash,
            "environment_hash": environment_matches_manifest,
            "protocol_hash": manifest.protocol_hash == PROTOCOL_HASH,
            "condition_hash": manifest.condition_hash
            == sha256_json([condition.to_dict() for condition in CONDITIONS]),
            "lexical_version": manifest.lexical_version == LEXICAL_VERSION,
            "judge_version": manifest.judge_version == JUDGE_VERSION,
            "analysis_version": manifest.analysis_version == ANALYSIS_VERSION,
            "smoke_profile_id": manifest.smoke_profile_id == SMOKE_PROFILE_ID,
        }
        if require_runtime_environment:
            checks["runtime_environment"] = (
                environment_matches_manifest and environment == runtime_environment()
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "workspace artifacts or software semantics changed: "
                + ", ".join(failed)
                + "; use a new workspace"
            )
        lexical = read_json(self.lexical_reference)
        if not isinstance(lexical, dict) or lexical.get("lexical_version") != LEXICAL_VERSION:
            raise RuntimeError("lexical-reference.json is missing or incompatible")
        return manifest

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Prevent concurrent mutation while recovering stale crash locks safely."""
        self.root.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        for _ in range(2):
            try:
                descriptor = os.open(self.lock_path, flags, 0o600)
                break
            except FileExistsError as exc:
                text = self.lock_path.read_text(encoding="utf-8", errors="replace")
                try:
                    pid = int(text.split("pid=", 1)[1].splitlines()[0])
                except (IndexError, ValueError):
                    pid = -1
                if not self._pid_alive(pid):
                    self.lock_path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"workspace is locked by process {pid}") from exc
        else:
            raise RuntimeError("could not acquire workspace lock")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
