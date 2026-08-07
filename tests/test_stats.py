from __future__ import annotations

import pytest

from instruction_duplication.json_types import is_object_sequence, object_value
from instruction_duplication.records import AnalysisRow
from instruction_duplication.stats import (
    DOUBLE,
    SINGLE,
    bootstrap_ci,
    build_analysis,
    contrast,
    holm_adjust,
    sign_flip_p_value,
)


def rows_with_failure():
    rows = []
    conditions = SINGLE + DOUBLE
    for qid in ("q1", "q2"):
        for condition in conditions:
            value = 1.0
            usable = True
            if qid == "q2" and condition == "system_before":
                value = 0.0
                usable = False
            rows.append(
                AnalysisRow(
                    question_id=qid,
                    model_id="m1",
                    dataset="d1",
                    condition_id=condition,
                    status="completed" if usable else "failed",
                    judgment={
                        "minimum_repair_scaffold": value,
                        "contrastive_scaffold_complete": value,
                        "facts_anchor_recall": value,
                        "shared_anchor_recall": value,
                        "firm_preprovisional_commitment": 0.0 if usable else 1.0,
                        "implications_anchor_recall": value,
                        "preanswer_anchor_recall": value,
                        "accuracy": value,
                        "generation_usable": usable,
                    },
                )
            )
    return rows


def test_itt_keeps_question_with_failed_treatment_cell():
    result = contrast(
        rows_with_failure(),
        "minimum_repair_scaffold",
        DOUBLE,
        SINGLE,
        permutations=100,
        bootstraps=100,
        confidence_level=0.95,
        seed=1,
    )
    assert result["question_clusters"] == 2
    assert result["treatment_mean"] == pytest.approx(5 / 6)
    assert result["control_mean"] == 1


def test_per_protocol_is_separate_and_shows_incompleteness():
    result = contrast(
        rows_with_failure(),
        "minimum_repair_scaffold",
        DOUBLE,
        SINGLE,
        permutations=100,
        bootstraps=100,
        confidence_level=0.95,
        seed=1,
        usable_only=True,
    )
    assert result["question_clusters"] == 1
    assert result["incomplete_model_question_units"] == 1


def test_pooled_endpoints_are_unadjusted_and_denominators_present():
    analysis = build_analysis(rows_with_failure(), permutations=100, bootstraps=100)
    for key in (
        "primary_repetition_contrasts",
        "trailing_copy_contrasts",
        "exploratory_contrasts",
        "per_protocol_sensitivity",
        "dataset_specific_contrasts",
    ):
        contrasts = analysis[key]
        assert is_object_sequence(contrasts) and contrasts
        assert all(
            "holm_p_value" not in object_value(item, name=f"{key} contrast") for item in contrasts
        )

    accuracy_contrast = object_value(analysis["accuracy_contrast"], name="accuracy contrast")
    assert "holm_p_value" not in accuracy_contrast

    primary = analysis["primary_repetition_contrasts"]
    assert is_object_sequence(primary)
    assert len(primary) == 5
    condition_means = object_value(analysis["condition_means"], name="condition_means")
    system = object_value(condition_means["system"], name="condition_means.system")
    accuracy = object_value(system["accuracy"], name="condition_means.system.accuracy")
    assert accuracy["n"] == 2


def test_model_specific_holm_is_scoped_within_each_endpoint():
    rows = rows_with_failure()
    rows.extend(
        AnalysisRow(
            question_id=row.question_id,
            model_id="m2",
            dataset=row.dataset,
            condition_id=row.condition_id,
            status=row.status,
            judgment=row.judgment,
        )
        for row in rows_with_failure()
    )
    analysis = build_analysis(rows, permutations=100, bootstraps=100)
    model_specific = analysis["model_specific_contrasts"]
    assert is_object_sequence(model_specific)
    items = [object_value(item, name="model-specific contrast") for item in model_specific]

    outcomes = {str(item["outcome"]) for item in items}
    for outcome in outcomes:
        family = [item for item in items if item["outcome"] == outcome]
        assert len(family) == 2
        expected = holm_adjust(
            [float(item["p_value"]) if item["p_value"] is not None else None for item in family]
        )
        actual = [
            float(item["holm_p_value"]) if item["holm_p_value"] is not None else None
            for item in family
        ]
        assert actual == expected


def test_holm_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_resampling_parameters_are_validated():
    with pytest.raises(ValueError):
        sign_flip_p_value([1], 0, 1)
    with pytest.raises(ValueError):
        bootstrap_ci([1], 0, 1)
    with pytest.raises(ValueError):
        bootstrap_ci([1], 10, 1, 1.5)
