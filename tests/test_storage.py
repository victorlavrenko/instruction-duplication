from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from instruction_duplication.lexical import LEXICAL_VERSION
from instruction_duplication.models import MODEL_BY_ID
from instruction_duplication.records import (
    AttemptFinished,
    AttemptRecord,
    CellCompletion,
    CellStarted,
    JudgmentWrite,
)
from instruction_duplication.storage import DATABASE_SCHEMA_VERSION, Database
from instruction_duplication.types import AttemptStatus, CellStatus


def test_legacy_database_is_rejected(tmp_path: Path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cells(id TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="predates schema versioning"):
        Database(path)


def test_database_constraints_plan_and_attempt_spend(tmp_path: Path, question):
    path = tmp_path / "run.sqlite3"
    model = MODEL_BY_ID["gemma-3-12b"]
    with Database(path) as db:
        assert db.prepare([question], [model.id]) == 8
        db.validate_plan([question.id], [model.id])
        cell = db.pending([model.id])[0]
        now = "2026-08-05T00:00:00+00:00"
        attempt = AttemptRecord(
            request_key="r1",
            phase="generation",
            cell_id=cell.cell_id,
            model_id=model.id,
            attempt_number=1,
            backend="fake",
            provider="fake",
            routed_model=model.id,
            status=AttemptStatus.COMPLETED,
            idempotency_key="i",
            requested_max_tokens=100,
            reservation_usd=0.02,
            reported_cost_usd=0.01,
            accounted_cost_usd=0.01,
            input_tokens=10,
            output_tokens=20,
            http_status=200,
            finish_reason="stop",
            latency_seconds=0.1,
            error=None,
            raw_response_json="{}",
            started_at=now,
            completed_at=now,
        )
        completion = CellCompletion(
            cell_id=cell.cell_id,
            status=CellStatus.COMPLETED,
            provider="fake",
            content="Final answer: A",
            error=None,
            input_tokens=10,
            output_tokens=20,
            latency_seconds=0.1,
            raw_response_json="{}",
            completed_at=now,
        )
        db.apply_generation_events(
            [
                CellStarted(cell_id=cell.cell_id, started_at=now),
                AttemptFinished(attempt=attempt, final=completion),
            ]
        )
        assert db.spend() == pytest.approx(0.01)
        assert db.reported_spend() == pytest.approx(0.01)
        assert db.counts()["completed"] == 1
        assert db.generation_attempt_state(cell.cell_id) == (1, "completed", 100)


def test_judgment_staleness_includes_status(tmp_path: Path, question):
    model = MODEL_BY_ID["gemma-3-12b"]
    with Database(tmp_path / "run.sqlite3") as db:
        db.prepare([question], [model.id])
        targets = db.judgment_targets(lexical_version=LEXICAL_VERSION)
        first = targets[0]
        now = "2026-08-05T00:00:00+00:00"
        db.put_judgments(
            [
                JudgmentWrite(
                    cell_id=str(first["cell_id"]),
                    content_hash=str(first["content_hash"]),
                    judged_at=now,
                    judgment={"x": 1},
                    question_id=question.id,
                )
            ],
            lexical_version=LEXICAL_VERSION,
        )
        assert len(db.judgment_targets(lexical_version=LEXICAL_VERSION)) == 7
        db.mark_running(first["cell_id"], now)
        # Same empty content but changed status must invalidate the judgment.
        ids = {row["cell_id"] for row in db.judgment_targets(lexical_version=LEXICAL_VERSION)}
        assert first["cell_id"] in ids


def test_schema_version_is_explicit(tmp_path: Path):
    with Database(tmp_path / "run.sqlite3") as db:
        row = db._conn.execute(
            "SELECT value FROM metadata WHERE key='database_schema_version'"
        ).fetchone()
        assert int(row["value"]) == DATABASE_SCHEMA_VERSION
