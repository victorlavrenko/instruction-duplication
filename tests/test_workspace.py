from __future__ import annotations

import json
from pathlib import Path

import pytest

from instruction_duplication.cli import main
from instruction_duplication.workspace import Workspace


def prepare(workspace: Path, local_jsonl: Path):
    main(
        [
            "prepare",
            "1",
            "--input-jsonl",
            str(local_jsonl),
            "--datasets",
            "test",
            "other",
            "--models",
            "gemma-3-12b",
            "--workspace",
            str(workspace),
        ]
    )


def test_manifest_detects_question_tampering(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    lines = ws.questions.read_text().splitlines()
    row = json.loads(lines[0])
    row["stem"] += " tampered"
    lines[0] = json.dumps(row)
    ws.questions.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="questions_hash"):
        ws.require_prepared()


def test_workspace_lock_rejects_live_process(tmp_path: Path):
    ws = Workspace(tmp_path)
    with ws.lock(), pytest.raises(RuntimeError, match="locked"), ws.lock():
        pass


def test_stale_lock_is_recovered(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.lock_path.write_text("pid=999999999\n")
    with ws.lock():
        assert ws.lock_path.exists()
    assert not ws.lock_path.exists()


def test_manifest_detects_package_version_change(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    manifest = json.loads(ws.manifest.read_text())
    manifest["package_version"] = "1.0.0"
    ws.manifest.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="package_version"):
        ws.require_prepared()


def test_manifest_detects_environment_change(tmp_path: Path, local_jsonl: Path):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    environment = json.loads(ws.environment.read_text())
    environment["python"] = "0.0.0"
    ws.environment.write_text(json.dumps(environment))
    with pytest.raises(RuntimeError, match="environment_hash"):
        ws.require_prepared()


def test_read_only_validation_does_not_require_generation_machine(
    monkeypatch, tmp_path: Path, local_jsonl: Path
):
    root = tmp_path / "run"
    prepare(root, local_jsonl)
    ws = Workspace(root)
    monkeypatch.setattr(
        "instruction_duplication.workspace.runtime_environment",
        lambda: {"python": "different"},
    )
    with pytest.raises(RuntimeError, match="runtime_environment"):
        ws.require_prepared()
    ws.require_prepared(require_runtime_environment=False)
