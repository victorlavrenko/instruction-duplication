"""Readable text rendering for the versioned statistical analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .json_types import is_object_sequence, number_value, object_value

LABELS = {
    "minimum_repair_scaffold": "Minimum strict repair scaffold",
    "contrastive_scaffold_complete": "Strict contrastive scaffold",
    "facts_anchor_recall": "Facts anchor recall",
    "shared_anchor_recall": "Facts/implications shared anchor recall",
    "firm_preprovisional_commitment": "Early explicit commitment (adverse)",
    "implications_anchor_recall": "Implications anchor recall",
    "preanswer_anchor_recall": "Combined pre-answer anchor recall",
    "accuracy": "Accuracy (ITT)",
}


def _value(value: object, digits: int = 4) -> str:
    return "NA" if value is None else f"{number_value(value, name='report value'):.{digits}f}"


def _p(value: object) -> str:
    if value is None:
        return "NA"
    number = number_value(value, name="p-value")
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def _object(value: object, name: str) -> Mapping[str, object]:
    return object_value(value, name=name)


def _object_list(value: object, name: str) -> list[Mapping[str, object]]:
    if not is_object_sequence(value):
        raise ValueError(f"{name} must be a list")
    return [_object(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _contrast_lines(item: Mapping[str, object], *, adjusted: bool = True) -> list[str]:
    outcome = str(item["outcome"])
    label = LABELS.get(outcome, outcome)
    confidence = int(number_value(item.get("confidence_level", 0.95), name="confidence") * 100)
    p_label = "raw p" if adjusted else "p"
    line = (
        f"  {label}: treatment={_value(item.get('treatment_mean'))}; "
        f"control={_value(item.get('control_mean'))}; effect={_value(item.get('effect'))}; "
        f"{confidence}% CI [{_value(item.get('ci_low'))}, {_value(item.get('ci_high'))}]; "
        f"{p_label}={_p(item.get('p_value'))}"
    )
    if adjusted:
        line += f"; Holm p={_p(item.get('holm_p_value'))}"
    line += (
        f"; question clusters={item.get('question_clusters', 0)}; "
        f"incomplete model-question units={item.get('incomplete_model_question_units', 0)}"
    )
    return [line]


def _section(
    title: str,
    items: Sequence[Mapping[str, object]],
    *,
    adjusted: bool = True,
) -> list[str]:
    lines = [title]
    if not items:
        lines.append("  No estimable contrasts.")
    for item in items:
        lines.extend(_contrast_lines(item, adjusted=adjusted))
    lines.append("")
    return lines


def render_report(analysis: Mapping[str, object]) -> str:
    """Render means, effects, denominators, and correction families."""
    sample = _object(analysis["sample"], "analysis.sample")
    status_counts = _object(sample["status_counts"], "analysis.sample.status_counts")
    estimand = _object(analysis["estimand"], "analysis.estimand")
    primary = _object_list(
        analysis["primary_repetition_contrasts"], "analysis.primary_repetition_contrasts"
    )
    trailing = _object_list(analysis["trailing_copy_contrasts"], "analysis.trailing_copy_contrasts")
    exploratory = _object_list(analysis["exploratory_contrasts"], "analysis.exploratory_contrasts")
    sensitivity = _object_list(
        analysis["per_protocol_sensitivity"], "analysis.per_protocol_sensitivity"
    )
    model_specific = _object_list(
        analysis["model_specific_contrasts"], "analysis.model_specific_contrasts"
    )
    dataset_specific = _object_list(
        analysis["dataset_specific_contrasts"], "analysis.dataset_specific_contrasts"
    )
    accuracy = _object(analysis["accuracy_contrast"], "analysis.accuracy_contrast")

    lines = [
        "INSTRUCTION DUPLICATION EXPERIMENT — VERSIONED ANALYSIS",
        "",
        "SAMPLE",
        f"  Models: {sample['models']}",
        f"  Datasets: {sample['datasets']}",
        f"  Questions: {sample['questions']}",
        f"  Planned cells: {sample['cells']}",
        f"  Usable generations: {sample['usable_cells']}",
        "  Status counts: " + ", ".join(f"{key}={value}" for key, value in status_counts.items()),
        "",
        "ESTIMAND",
        f"  {estimand['primary']}",
        f"  {estimand['failure_policy']}",
        "",
    ]
    lines += _section(
        "PRIMARY REPETITION CONTRASTS (pre-specified endpoints; unadjusted)",
        primary,
        adjusted=False,
    )
    lines += _section(
        "TRAILING-COPY CONTRASTS (exploratory; unadjusted)",
        trailing,
        adjusted=False,
    )
    lines += _section(
        "EXPLORATORY DIAGNOSTICS (unadjusted; not confirmatory)",
        exploratory,
        adjusted=False,
    )
    lines += ["ACCURACY", *_contrast_lines(accuracy, adjusted=False), ""]
    lines += _section(
        "PER-PROTOCOL SENSITIVITY (successful generations only; unadjusted)",
        sensitivity,
        adjusted=False,
    )

    lines.append("MODEL-SPECIFIC HETEROGENEITY (Holm-adjusted across models within endpoint)")
    for item in model_specific:
        lines.append(f"  [{item['model_id']}] " + _contrast_lines(item)[0].strip())
    lines.append("")
    lines.append("DATASET-SPECIFIC HETEROGENEITY (unadjusted; descriptive)")
    for item in dataset_specific:
        lines.append(f"  [{item['dataset']}] " + _contrast_lines(item, adjusted=False)[0].strip())
    lines.append("")

    condition_means = _object(analysis["condition_means"], "analysis.condition_means")
    lines.append("CONDITION MEANS (mean; n)")
    for condition, raw_metrics in condition_means.items():
        metrics = _object(raw_metrics, f"analysis.condition_means.{condition}")
        rendered_parts: list[str] = []
        for outcome, raw_values in metrics.items():
            values = _object(raw_values, f"analysis.condition_means.{condition}.{outcome}")
            rendered_parts.append(
                f"{LABELS.get(outcome, outcome)}={_value(values['mean'])};n={values['n']}"
            )
        lines.append(f"  {condition}: {', '.join(rendered_parts)}")
    lines.append("")
    lines.append(
        "Lexical anchor outcomes are deterministic surface-coverage diagnostics. "
        "They do not establish semantic equivalence, medical validity, or faithful "
        "internal reasoning."
    )
    return "\n".join(lines).rstrip() + "\n"
