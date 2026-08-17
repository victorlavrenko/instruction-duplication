"""Reproducibility snapshots and blinded human-validation exports."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .io_utils import sha256_json, write_json, write_jsonl
from .json_types import JsonObject, json_object, object_value
from .models import Model

MODEL_ELIGIBILITY_VERSION = "single-visible-stream-policy-v2"
HUMAN_AUDIT_VERSION = "blinded-matched-pairs-v2"
AUDIT_PAIR_LIMIT = 200

PAIRINGS: tuple[tuple[str, str, str], ...] = (
    ("system", "system_before", "add_before_copy"),
    ("system", "system_after", "add_after_copy"),
    ("before", "system_before", "add_system_copy"),
    ("before", "before_after", "add_after_copy"),
    ("after", "system_after", "add_system_copy"),
    ("after", "before_after", "add_before_copy"),
)


def model_eligibility_snapshot(models: Sequence[Model]) -> JsonObject:
    """Freeze the model-panel basis without asserting unverifiable architecture facts."""
    entries: list[JsonObject] = [
        {
            "model_id": model.id,
            "huggingface_checkpoint": model.hf_id,
            "openrouter_model": model.openrouter_id,
            "configuration_sha256": sha256_json(model.to_dict()),
            "eligibility_basis": [
                "panel entry is an instruction-tuned checkpoint",
                "requests do not enable a separate reasoning channel",
                "preflight and generation reject exposed reasoning/thinking fields",
                "only visible assistant content is stored and judged",
            ],
            "catalog_claims_about_hidden_architecture": None,
        }
        for model in models
    ]
    payload = json_object(
        {
            "model_eligibility_version": MODEL_ELIGIBILITY_VERSION,
            "policy": (
                "single visible response stream; no separately exposed reasoning channel; "
                "provider route is pinned only after two successful guarded probes"
            ),
            "models": entries,
        },
        path="model eligibility snapshot",
    )
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload


def human_audit_schema() -> JsonObject:
    """Return the annotation contract stored beside the blinded pairs."""
    return {
        "human_audit_version": HUMAN_AUDIT_VERSION,
        "instructions": (
            "Rate each response independently before choosing between them. Do not open the "
            "key until all ratings are frozen. Judge whether the visible text actually covers "
            "the stem facts and qualifiers; do not infer credit from marker presence alone."
        ),
        "ordinal_scale": {
            "0": "absent or materially wrong",
            "1": "partial or ambiguous",
            "2": "clear and adequate",
        },
        "response_rating_fields": {
            "final_answer_clear": "boolean or null",
            "trajectory_complete": "0, 1, 2, or null",
            "important_facts_covered": "0, 1, 2, or null",
            "qualifiers_preserved": "0, 1, 2, or null",
            "facts_traced_to_implications": "0, 1, 2, or null",
            "contrastive_check_substantive": "0, 1, 2, or null",
            "overall_repair_scaffold": "0, 1, 2, or null",
            "notes": "free text or null",
        },
        "pair_fields": {
            "preferred_response": "A, B, tie, or null",
            "preference_reason": "free text or null",
            "reviewer_id": "pseudonymous reviewer id or null",
        },
    }


def _pair_hash(*parts: str) -> str:
    return hashlib.sha256((HUMAN_AUDIT_VERSION + "\0" + "\0".join(parts)).encode()).hexdigest()


def _response_payload(row: Mapping[str, object]) -> JsonObject:
    return {
        "status": str(row.get("status", "")),
        "response": None if row.get("content") is None else str(row["content"]),
        "ratings": {
            "final_answer_clear": None,
            "trajectory_complete": None,
            "important_facts_covered": None,
            "qualifiers_preserved": None,
            "facts_traced_to_implications": None,
            "contrastive_check_substantive": None,
            "overall_repair_scaffold": None,
            "notes": None,
        },
    }


def export_blinded_matched_pairs(
    rows: Iterable[JsonObject],
    *,
    audit_path: Path,
    key_path: Path,
    schema_path: Path,
    limit: int = AUDIT_PAIR_LIMIT,
) -> JsonObject:
    """Write a deterministic score-independent sample and its separate condition key."""
    if limit < 1:
        raise ValueError("human audit limit must be positive")
    by_unit: dict[tuple[str, str], dict[str, JsonObject]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        model_id = str(row["model_id"])
        by_unit.setdefault((question_id, model_id), {})[str(row["condition_id"])] = row

    candidates: list[tuple[str, JsonObject, JsonObject]] = []
    for (question_id, model_id), conditions in by_unit.items():
        for control, treatment, added_copy in PAIRINGS:
            if control not in conditions or treatment not in conditions:
                continue
            left = conditions[control]
            right = conditions[treatment]
            pair_id = _pair_hash(question_id, model_id, control, treatment)[:20]
            treatment_first = int(pair_id[-1], 16) % 2 == 0
            first, second = (right, left) if treatment_first else (left, right)
            blind_row = json_object(
                {
                    "human_audit_version": HUMAN_AUDIT_VERSION,
                    "pair_id": pair_id,
                    "question": str(left["stem"]),
                    "choices": object_value(left["choices"], name="audit choices"),
                    "response_a": _response_payload(first),
                    "response_b": _response_payload(second),
                    "preferred_response": None,
                    "preference_reason": None,
                    "reviewer_id": None,
                },
                path=f"blinded audit pair {pair_id}",
            )
            key_row = json_object(
                {
                    "pair_id": pair_id,
                    "question_id": question_id,
                    "model_id": model_id,
                    "dataset": str(left["dataset"]),
                    "added_copy": added_copy,
                    "response_a_condition": str(first["condition_id"]),
                    "response_b_condition": str(second["condition_id"]),
                    "treatment_response": "A" if treatment_first else "B",
                },
                path=f"audit key {pair_id}",
            )
            candidates.append((_pair_hash(pair_id, "sample"), blind_row, key_row))
    candidates.sort(key=lambda item: item[0])
    selected = candidates[: min(limit, len(candidates))]
    audit_rows = [item[1] for item in selected]
    key_rows = [item[2] for item in selected]
    write_jsonl(audit_path, audit_rows)
    write_jsonl(key_path, key_rows)
    write_json(schema_path, human_audit_schema())
    return json_object(
        {
            "human_audit_version": HUMAN_AUDIT_VERSION,
            "candidate_pairs": len(candidates),
            "exported_pairs": len(selected),
            "selection": (
                "lowest versioned SHA-256 pair hashes; independent of responses, conditions "
                "after blinding, and mechanical scores"
            ),
            "condition_key_separate": True,
        },
        path="human audit metadata",
    )
