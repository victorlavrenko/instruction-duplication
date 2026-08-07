from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from instruction_duplication.datasets_loader import (
    DATASET_SPECS,
    DatasetNormalizationError,
    load_hf_dataset,
    normalize_row,
    select_questions,
)
from instruction_duplication.json_types import object_value


def base_row():
    return {
        "id": "1",
        "question": "A patient has sudden hearing loss. What is the diagnosis?",
        "options": {"A": "SSNHL", "B": "Cerumen"},
        "answer_idx": "0",
    }


def test_malformed_earlier_gold_field_does_not_block_valid_later_field():
    row = base_row() | {"answer_idx": "bad", "correct_answer": "A"}
    question = normalize_row(row, dataset="d", source_split="test", index=0)
    assert question.gold == "A"
    assert question.gold_source == "correct_answer"


def test_conflicting_gold_fields_are_rejected():
    row = base_row() | {"correct_answer": "B"}
    with pytest.raises(DatasetNormalizationError, match="conflicting gold"):
        normalize_row(row, dataset="d", source_split="test", index=0)


def test_conflicting_choice_encodings_are_rejected():
    row = base_row() | {"choices": {"A": "Different", "B": "Cerumen"}}
    with pytest.raises(DatasetNormalizationError, match="conflicting choice"):
        normalize_row(row, dataset="d", source_split="test", index=0)


@pytest.mark.parametrize(
    "stem",
    [
        "The radiograph demonstrates a lesion. What is it?",
        "In the following table, which value is abnormal?",
        "The ECG shown below indicates what diagnosis?",
    ],
)
def test_media_dependent_questions_are_rejected(stem):
    row = base_row() | {"question": stem}
    with pytest.raises(DatasetNormalizationError, match="media"):
        normalize_row(row, dataset="d", source_split="test", index=0)


def test_local_selection_filters_deduplicates_and_uses_seed(tmp_path: Path):
    rows = []
    for dataset in ("d1", "d2"):
        rows.extend(
            {
                "id": f"{dataset}-{index}",
                "dataset": dataset,
                "stem": f"Unique {dataset} stem {index}",
                "choices": {"A": "Yes", "B": "No"},
                "gold": "A",
                "gold_text": "Yes",
                "gold_source": "fixture",
                "gold_raw": "A",
                "source_split": "test",
            }
            for index in range(3)
        )
        rows.append(rows[-1] | {"id": f"{dataset}-duplicate"})
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first, audit = select_questions(
        dataset_names=["d1", "d2"], questions_per_dataset=2, seed=7, input_jsonl=path
    )
    second, _ = select_questions(
        dataset_names=["d1", "d2"], questions_per_dataset=2, seed=7, input_jsonl=path
    )
    assert [q.id for q in first] == [q.id for q in second]
    assert len(first) == 4
    assert object_value(audit["selection"], name="audit.selection")["seed"] == 7


def test_pinned_hf_source_is_passed_without_fallback(monkeypatch):
    calls = []

    def fake_load_dataset(repository, config, **kwargs):
        calls.append((repository, config, kwargs))
        return [base_row()]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    questions, audit = load_hf_dataset("medqa")
    spec = DATASET_SPECS["medqa"]
    assert len(questions) == 1
    assert calls == [
        (spec.repository, spec.config, {"split": spec.split, "revision": spec.revision})
    ]
    assert object_value(audit["source"], name="audit.source")["revision"] == spec.revision


def test_all_dataset_revisions_are_full_hashes():
    assert all(len(spec.revision) == 40 for spec in DATASET_SPECS.values())


def test_afrimedqa_string_options_and_clean_question_are_normalized():
    row = {
        "sample_id": "sample-1",
        "split": "test",
        "question_type": "MCQ",
        "question": "uncleanquestion",
        "question_clean": "Clean question?",
        "answer_options": "{'option1': 'First', 'option2': 'Second', 'option3': 'Third'}",
        "correct_answer": "option2",
    }
    question = normalize_row(
        row,
        dataset="afrimedqa",
        source_split="internal test partition",
        index=0,
    )
    assert question.id == "afrimedqa:sample-1"
    assert question.stem == "Clean question?"
    assert dict(question.choices) == {"A": "First", "B": "Second", "C": "Third"}
    assert question.gold == "B"
