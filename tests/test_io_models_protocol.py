from __future__ import annotations

import json
from pathlib import Path

import pytest

from instruction_duplication.io_utils import canonical_json, write_json, write_jsonl
from instruction_duplication.models import MODEL_BY_ID, Model, select_models
from instruction_duplication.protocol import PROTOCOL, render_messages


def test_strict_json_converts_nonfinite_to_null(tmp_path: Path):
    assert canonical_json({"x": float("nan")}) == '{"x":null}'
    path = tmp_path / "x.json"
    write_json(path, {"x": float("inf")})
    assert json.loads(path.read_text()) == {"x": None}


def test_jsonl_stream_write(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, ({"i": index} for index in range(3)))
    assert path.read_text().count("\n") == 3


def test_question_choices_are_immutable(question):
    with pytest.raises(TypeError):
        question.choices["A"] = "changed"


def test_model_validation_and_order():
    selected = select_models(["qwen3-30b-a3b-instruct-2507", "gemma-3-12b"])
    assert [model.id for model in selected] == [
        "qwen3-30b-a3b-instruct-2507",
        "gemma-3-12b",
    ]
    with pytest.raises(ValueError):
        select_models(["gemma-3-12b", "gemma-3-12b"])
    row = MODEL_BY_ID["gemma-3-12b"].to_dict()
    assert row["preferred_backend"] == "huggingface"
    row["max_concurrency"] = 0
    with pytest.raises(ValueError):
        Model.from_dict(row)
    row = MODEL_BY_ID["gemma-3-12b"].to_dict()
    row["preferred_backend"] = "invalid"
    with pytest.raises(ValueError):
        Model.from_dict(row)


def test_qwen3_235b_prefers_openrouter_away_from_deepinfra():
    model = MODEL_BY_ID["qwen3-235b-a22b-instruct-2507"]
    assert model.preferred_backend == "openrouter"
    assert model.openrouter_providers == ("alibaba", "parasail", "novita", "deepinfra")


def test_protocol_is_single_authority_and_escapes_xml(question):
    messages = render_messages(
        question.stem + " & value < 3",
        question.choices,
        "system_before_after",
    )
    assert messages[0]["content"].count(PROTOCOL) == 1
    assert messages[1]["content"].count(PROTOCOL) == 2
    assert "&amp;" in messages[1]["content"]
    assert "&lt;" in messages[1]["content"]


def test_smoke_profile_matches_model_request_limits():
    import json
    from importlib.resources import files

    from instruction_duplication.models import MODELS, SMOKE_PROFILE_ID

    profile = json.loads(
        files("instruction_duplication")
        .joinpath("smoke-profile-2026-08-05.json")
        .read_text(encoding="utf-8")
    )
    assert profile["profile_id"] == SMOKE_PROFILE_ID
    for model in MODELS:
        row = profile["models"][model.id]
        assert row["output_tokens"]["max"] == model.smoke_observed_max_output_tokens
        assert row["request_output_tokens"] == model.request_output_tokens
        assert model.request_output_tokens >= 1.25 * model.smoke_observed_max_output_tokens
