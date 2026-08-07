"""Versioned paired ITT analysis with endpoint-scoped multiplicity control."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from . import __version__
from .json_types import JsonObject, json_object
from .manifest import ANALYSIS_VERSION
from .records import AnalysisRow

SINGLE = ("system", "before", "after")
DOUBLE = ("system_before", "system_after", "before_after")
TRAILING_TREATMENT = ("system_after", "before_after")
TRAILING_CONTROL = ("system", "before")

# Confirmatory outcomes are reported individually; no arbitrary weighted composite is used.
PRIMARY_OUTCOMES = (
    "minimum_repair_scaffold",
    "contrastive_scaffold_complete",
    "facts_anchor_recall",
    "shared_anchor_recall",
    "firm_preprovisional_commitment",
)
TRAILING_OUTCOMES = (
    "minimum_repair_scaffold",
    "contrastive_scaffold_complete",
    "facts_anchor_recall",
    "shared_anchor_recall",
)
MODEL_OUTCOMES = PRIMARY_OUTCOMES
EXPLORATORY_OUTCOMES = (
    "implications_anchor_recall",
    "preanswer_anchor_recall",
)


def _numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def sign_flip_p_value(values: Sequence[float], iterations: int, seed: int) -> float | None:
    """Return a two-sided paired sign-flip p-value, exact for small samples."""
    if iterations < 1:
        raise ValueError("permutations must be positive")
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    if observed < 1e-15:
        return 1.0
    if len(values) <= 16:
        draws = itertools.product((-1.0, 1.0), repeat=len(values))
        extreme = 0
        total = 0
        for signs in draws:
            total += 1
            signed_sum = sum(sign * value for sign, value in zip(signs, values, strict=True))
            statistic = abs(signed_sum / len(values))
            if statistic >= observed - 1e-15:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        statistic = abs(
            sum(value if rng.random() < 0.5 else -value for value in values) / len(values)
        )
        if statistic >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def bootstrap_ci(
    values: Sequence[float],
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return a linear-interpolated percentile bootstrap interval by question."""
    if iterations < 1:
        raise ValueError("bootstraps must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    alpha = (1 - confidence_level) / 2
    return _quantile(draws, alpha), _quantile(draws, 1 - alpha)


def _judgment_value(judgment: Mapping[str, object], key: str) -> object:
    return judgment.get(key)


def _unit_values(
    rows: Iterable[AnalysisRow],
    outcome: str,
    *,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> dict[tuple[str, str], dict[str, float]]:
    units: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if model_id is not None and row.model_id != model_id:
            continue
        if dataset is not None and row.dataset != dataset:
            continue
        judgment = row.judgment or {}
        if usable_only and not bool(_judgment_value(judgment, "generation_usable")):
            continue
        value = _numeric(_judgment_value(judgment, outcome))
        if value is None:
            continue
        units[(row.question_id, row.model_id)][row.condition_id] = value
    return units


def _paired_data(
    rows: Iterable[AnalysisRow],
    outcome: str,
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> tuple[list[float], list[float], list[float], int]:
    units = _unit_values(
        rows,
        outcome,
        model_id=model_id,
        dataset=dataset,
        usable_only=usable_only,
    )
    by_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
    required = tuple(treatment) + tuple(control)
    incomplete = 0
    for (question_id, _model), values in units.items():
        if not all(condition in values for condition in required):
            incomplete += 1
            continue
        treated = sum(values[condition] for condition in treatment) / len(treatment)
        baseline = sum(values[condition] for condition in control) / len(control)
        by_question[question_id].append((treated, baseline))
    treatment_means: list[float] = []
    control_means: list[float] = []
    differences: list[float] = []
    for pairs in by_question.values():
        treated = sum(value[0] for value in pairs) / len(pairs)
        baseline = sum(value[1] for value in pairs) / len(pairs)
        treatment_means.append(treated)
        control_means.append(baseline)
        differences.append(treated - baseline)
    return treatment_means, control_means, differences, incomplete


def contrast(
    rows: Iterable[AnalysisRow],
    outcome: str,
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed: int,
    model_id: str | None = None,
    dataset: str | None = None,
    usable_only: bool = False,
) -> JsonObject:
    """Compute one question-clustered paired contrast with visible denominators."""
    treated, baseline, differences, incomplete = _paired_data(
        rows,
        outcome,
        treatment,
        control,
        model_id=model_id,
        dataset=dataset,
        usable_only=usable_only,
    )
    low, high = bootstrap_ci(differences, bootstraps, seed + 1, confidence_level)
    result: JsonObject = {
        "outcome": outcome,
        "estimand": "question-clustered mean paired difference",
        "population": "per-protocol sensitivity" if usable_only else "intention-to-treat",
        "treatment_conditions": list(treatment),
        "control_conditions": list(control),
        "treatment_mean": _mean(treated),
        "control_mean": _mean(baseline),
        "effect": _mean(differences),
        "ci_low": low,
        "ci_high": high,
        "confidence_level": confidence_level,
        "p_value": sign_flip_p_value(differences, permutations, seed + 2),
        "question_clusters": len(differences),
        "incomplete_model_question_units": incomplete,
    }
    if model_id is not None:
        result["model_id"] = model_id
    if dataset is not None:
        result["dataset"] = dataset
    return result


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Apply Holm step-down correction to one explicitly declared family."""
    adjusted: list[float | None] = [None] * len(p_values)
    finite = sorted(
        ((value, index) for index, value in enumerate(p_values) if value is not None),
        key=lambda item: item[0],
    )
    running = 0.0
    count = len(finite)
    for rank, (value, index) in enumerate(finite):
        running = max(running, (count - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def _adjust_family(results: Sequence[JsonObject], field: str = "holm_p_value") -> None:
    adjusted = holm_adjust([_numeric(item.get("p_value")) for item in results])
    for item, value in zip(results, adjusted, strict=True):
        item[field] = value


def _condition_means(rows: Sequence[AnalysisRow]) -> JsonObject:
    outcomes = PRIMARY_OUTCOMES + EXPLORATORY_OUTCOMES + ("accuracy", "generation_usable")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        judgment = row.judgment or {}
        for outcome in outcomes:
            value = _numeric(_judgment_value(judgment, outcome))
            if value is not None:
                grouped[row.condition_id][outcome].append(value)
    return {
        condition: {
            outcome: {"mean": _mean(values), "n": len(values)}
            for outcome, values in metrics.items()
        }
        for condition, metrics in grouped.items()
    }


def _outcome_contrasts(
    rows: Sequence[AnalysisRow],
    outcomes: Sequence[str],
    treatment: Sequence[str],
    control: Sequence[str],
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed_base: int,
    usable_only: bool = False,
    model_id: str | None = None,
    dataset: str | None = None,
) -> list[JsonObject]:
    results = [
        contrast(
            rows,
            outcome,
            treatment,
            control,
            permutations=permutations,
            bootstraps=bootstraps,
            confidence_level=confidence_level,
            seed=seed_base + index * 100,
            usable_only=usable_only,
            model_id=model_id,
            dataset=dataset,
        )
        for index, outcome in enumerate(outcomes)
    ]
    return results


def _heterogeneity_contrasts(
    rows: Sequence[AnalysisRow],
    groups: Sequence[str],
    *,
    group_field: Literal["model_id", "dataset"],
    outcomes: Sequence[str],
    permutations: int,
    bootstraps: int,
    confidence_level: float,
    seed_base: int,
) -> list[JsonObject]:
    results: list[JsonObject] = []
    for group_index, group in enumerate(groups):
        for outcome_index, outcome in enumerate(outcomes):
            results.append(
                contrast(
                    rows,
                    outcome,
                    DOUBLE,
                    SINGLE,
                    permutations=permutations,
                    bootstraps=bootstraps,
                    confidence_level=confidence_level,
                    seed=seed_base + group_index * 1_000 + outcome_index * 100,
                    model_id=group if group_field == "model_id" else None,
                    dataset=group if group_field == "dataset" else None,
                )
            )
    return results


def _adjust_within_outcome(results: Sequence[JsonObject]) -> None:
    """Apply Holm separately across groups for each reported outcome."""
    by_outcome: dict[str, list[JsonObject]] = defaultdict(list)
    for result in results:
        by_outcome[str(result["outcome"])].append(result)
    for family in by_outcome.values():
        _adjust_family(family)


def _sample_summary(
    rows: Sequence[AnalysisRow],
    models: Sequence[str],
    datasets: Sequence[str],
) -> JsonObject:
    status_counts: dict[str, int] = defaultdict(int)
    usable = 0
    for row in rows:
        status_counts[row.status] += 1
        judgment = row.judgment or {}
        usable += int(bool(_judgment_value(judgment, "generation_usable")))
    return {
        "models": len(models),
        "datasets": len(datasets),
        "questions": len({row.question_id for row in rows}),
        "cells": len(rows),
        "usable_cells": usable,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _analysis_metadata(
    permutations: int,
    bootstraps: int,
    confidence_level: float,
) -> JsonObject:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "package_version": __version__,
        "status": "versioned deterministic reanalysis",
        "parameters": {
            "permutations": permutations,
            "bootstraps": bootstraps,
            "confidence_level": confidence_level,
            "seeds": {
                "primary_base": 13_000,
                "trailing_base": 16_000,
                "exploratory_base": 20_000,
                "accuracy": 21_100,
                "sensitivity_base": 23_000,
                "model_base": 30_000,
                "dataset_base": 50_000,
            },
        },
        "estimand": {
            "primary": (
                "mean of three two-copy conditions minus mean of three one-copy "
                "conditions, paired within model-question then equally clustered by question"
            ),
            "failure_policy": (
                "ITT: instructed generation failures score 0 on beneficial repair outcomes "
                "and 1 on adverse early commitment; accuracy failures score 0"
            ),
            "sensitivity": "complete successful generations only, reported separately",
        },
        "multiplicity_families": {
            "primary": "none; pre-specified pooled endpoints are reported individually",
            "trailing": "none; trailing-copy placement contrasts are exploratory",
            "per_protocol": "none; sensitivity endpoints mirror the pooled primary endpoints",
            "model_specific": "Holm across models separately within each endpoint",
            "dataset_specific": "none; dataset heterogeneity is descriptive",
            "exploratory": "none; exploratory diagnostics are explicitly unadjusted",
            "accuracy": "none; accuracy is reported as a separate performance endpoint",
        },
    }


def build_analysis(
    rows: Sequence[AnalysisRow],
    *,
    permutations: int = 50_000,
    bootstraps: int = 10_000,
    confidence_level: float = 0.95,
) -> JsonObject:
    """Build ITT contrasts, sensitivity analyses, and heterogeneity tables."""
    if permutations < 1 or bootstraps < 1:
        raise ValueError("permutations and bootstraps must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    primary = _outcome_contrasts(
        rows,
        PRIMARY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=13_000,
    )
    trailing = _outcome_contrasts(
        rows,
        TRAILING_OUTCOMES,
        TRAILING_TREATMENT,
        TRAILING_CONTROL,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=16_000,
    )
    exploratory = _outcome_contrasts(
        rows,
        EXPLORATORY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=20_000,
    )
    accuracy = contrast(
        rows,
        "accuracy",
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed=21_100,
    )
    sensitivity = _outcome_contrasts(
        rows,
        PRIMARY_OUTCOMES,
        DOUBLE,
        SINGLE,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=23_000,
        usable_only=True,
    )
    models = sorted({row.model_id for row in rows})
    datasets = sorted({row.dataset for row in rows})
    model_specific = _heterogeneity_contrasts(
        rows,
        models,
        group_field="model_id",
        outcomes=MODEL_OUTCOMES,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=30_000,
    )
    _adjust_within_outcome(model_specific)
    dataset_specific = _heterogeneity_contrasts(
        rows,
        datasets,
        group_field="dataset",
        outcomes=PRIMARY_OUTCOMES,
        permutations=permutations,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
        seed_base=50_000,
    )
    return json_object(
        {
            **_analysis_metadata(permutations, bootstraps, confidence_level),
            "primary_repetition_contrasts": primary,
            "accuracy_contrast": accuracy,
            "trailing_copy_contrasts": trailing,
            "exploratory_contrasts": exploratory,
            "per_protocol_sensitivity": sensitivity,
            "model_specific_contrasts": model_specific,
            "dataset_specific_contrasts": dataset_specific,
            "condition_means": _condition_means(rows),
            "sample": _sample_summary(rows, models, datasets),
        },
        path="analysis",
    )
