"""Deterministic stem-anchor coverage diagnostics."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping

from .json_types import JsonObject
from .types import Question

LEXICAL_VERSION = "unicode-polarity-anchor-recall-v2"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
    "would",
    "patient",
    "patients",
    "following",
    "most",
    "likely",
    "best",
    "answer",
    "diagnosis",
    "next",
    "step",
    "question",
}
NEGATIONS = {"no", "not", "never", "without", "denies", "denied", "absent", "absence"}
LATERALITY = {"left", "right", "bilateral", "ipsilateral", "contralateral"}
TIMING = {
    "acute",
    "chronic",
    "sudden",
    "gradual",
    "hours",
    "hour",
    "days",
    "day",
    "weeks",
    "week",
    "months",
    "month",
    "years",
    "year",
}
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|\d+(?:\.\d+)?%?", re.UNICODE)


def normalize_token(token: str) -> str:
    """Apply conservative Unicode normalization without heuristic stemming."""
    return unicodedata.normalize("NFKC", token).casefold().strip("-'_")


def content_tokens(text: str) -> list[str]:
    """Return Unicode-aware non-stopword tokens."""
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = normalize_token(raw)
        if len(token) > 1 and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def anchor_tokens(text: str) -> list[str]:
    """Return tokens plus local polarity anchors for deterministic fact matching."""
    tokens = content_tokens(text)
    anchors = list(tokens)
    for index, token in enumerate(tokens):
        if token in NEGATIONS:
            anchors.extend(f"NEG::{following}" for following in tokens[index + 1 : index + 4])
    return anchors


def _weight(token: str) -> float:
    plain = token.removeprefix("NEG::")
    if token.startswith("NEG::") or plain in NEGATIONS or plain.replace(".", "").isdigit():
        return 2.0
    if plain in LATERALITY or plain in TIMING or plain.endswith("%"):
        return 1.5
    return 1.0


def build_reference(questions: Iterable[Question]) -> JsonObject:
    """Build a versioned rule manifest; scores do not depend on the selected sample."""
    question_count = sum(1 for _ in questions)
    return {
        "lexical_version": LEXICAL_VERSION,
        "question_count_for_audit_only": question_count,
        "weighting": {
            "ordinary": 1.0,
            "laterality_or_timing": 1.5,
            "numeric_or_polarity": 2.0,
        },
        "sample_dependent_idf": False,
    }


def _validate_reference(reference: Mapping[str, object]) -> None:
    if reference.get("lexical_version") != LEXICAL_VERSION:
        raise ValueError("lexical reference version does not match the current scorer")


def _recall(stem: str, output: str) -> float:
    stem_counts = Counter(anchor_tokens(stem))
    output_counts = Counter(anchor_tokens(output))
    total = 0.0
    credited = 0.0
    for token, frequency in stem_counts.items():
        weight = _weight(token) * frequency
        total += weight
        credited += _weight(token) * min(frequency, output_counts.get(token, 0))
    return credited / total if total else 0.0


def _shared_recall(stem: str, facts: str, implications: str) -> float:
    stem_counts = Counter(anchor_tokens(stem))
    facts_counts = Counter(anchor_tokens(facts))
    implications_counts = Counter(anchor_tokens(implications))
    total = 0.0
    credited = 0.0
    for token, frequency in stem_counts.items():
        total += _weight(token) * frequency
        shared = min(frequency, facts_counts.get(token, 0), implications_counts.get(token, 0))
        credited += _weight(token) * shared
    return credited / total if total else 0.0


def score_preanswer(
    stem: str,
    facts: str,
    implications: str,
    reference: Mapping[str, object],
) -> dict[str, float]:
    """Measure literal, polarity-aware stem-anchor coverage before commitment."""
    _validate_reference(reference)
    return {
        "facts_anchor_recall": _recall(stem, facts),
        "implications_anchor_recall": _recall(stem, implications),
        "preanswer_anchor_recall": _recall(stem, f"{facts} {implications}"),
        "shared_anchor_recall": _shared_recall(stem, facts, implications),
    }
