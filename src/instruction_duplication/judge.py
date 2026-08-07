"""Strict XML parsing and deterministic repair-oriented measurements."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

from .answer_utils import ExtractedAnswer, extract_answer, normalize_text
from .lexical import content_tokens, score_preanswer
from .manifest import JUDGE_VERSION
from .types import CellStatus, Question


class CommitmentEvent(TypedDict):
    """One literal answer commitment detected before the provisional answer."""

    option: str
    start: int
    evidence: str


class Judgment(TypedDict):
    """Typed deterministic judgment payload shared with storage and analysis."""

    judge_version: str
    generation_usable: bool
    generation_status: str
    minimum_repair_scaffold: float | None
    contrastive_scaffold_complete: float | None
    facts_anchor_recall: float | None
    implications_anchor_recall: float | None
    preanswer_anchor_recall: float | None
    shared_anchor_recall: float | None
    firm_preprovisional_commitment: float | None
    xml_document_valid: bool | None
    protocol_complete: bool | None
    accuracy: int
    accuracy_parseable: bool
    final_option: str | None
    answer_extraction_status: str
    parser_errors: list[str]
    answer_candidates: list[str]
    repair_readiness_criteria: dict[str, bool]
    section_substantive: dict[str, bool]
    section_token_counts: dict[str, int]
    provisional_option: str | None
    second_best_option: str | None
    rereasoning_decision: str | None
    decision_consistent: bool
    early_commitments: list[CommitmentEvent]


ROOT_CHILDREN = (
    "facts",
    "implications",
    "provisional_answer",
    "contrastive_check",
    "rereasoning",
    "final_answer",
)
CONTRASTIVE_CHILDREN = ("second_best", "decisive_fact", "answer_changing_change")
VALID_DECISIONS = {"retain", "revise"}
COUNTERFACTUAL_CUES = re.compile(
    r"\b(?:if|were|instead|without|absence|presence|remove|add|replace|change|"
    r"changed|would|become)\b",
    re.IGNORECASE,
)
OPTION_ATTR_RE = re.compile(r"\boption\s*=\s*['\"]\s*([A-Z])\s*['\"]", re.IGNORECASE)
FINAL_TAG_RE = re.compile(
    r"<final_answer(?P<attrs>\s+[^>]*)?>(?P<body>.*?)</final_answer>",
    re.IGNORECASE | re.DOTALL,
)
XML_FENCE_RE = re.compile(
    r"\A```xml\r?\n(?P<document>.*)\r?\n```\Z",
    re.IGNORECASE | re.DOTALL,
)
EXPLICIT_LABEL_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|choice|option|diagnosis|treatment|management|cause)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"option\s*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)
BARE_LABEL_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+|my\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|choice|option|diagnosis|treatment|management|cause)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"\(?([A-Z])\)?\b"
)
TEXT_COMMITMENT_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+|correct\s+|best\s+|most\s+likely\s+|provisional\s+)?"
    r"(?:answer|diagnosis|treatment|management|cause)\s*"
    r"(?:is|would\s+be|should\s+be|must\s+be|appears\s+to\s+be|seems\s+to\s+be|=|:)\s*"
    r"(.+?)(?:[.;\n<]|$)",
    re.IGNORECASE,
)
COMMITMENT_EXCLUSION = re.compile(
    r"\b(?:not|incorrect|wrong|false|less\s+likely|unlikely|argues?\s+against|rules?\s+out|"
    r"second[- ]best|alternative|differential|hypothetical|could|may|might|only\s+if|unless)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedProtocol:
    """A strict protocol parse with audit diagnostics."""

    document: str
    xml_valid: bool
    structure_valid: bool
    errors: tuple[str, ...]
    sections: Mapping[str, str]
    contrast_sections: Mapping[str, str]
    provisional_option: str | None
    second_best_option: str | None
    final_option: str | None
    decision: str | None
    provisional_start: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", MappingProxyType(dict(self.sections)))
        object.__setattr__(
            self,
            "contrast_sections",
            MappingProxyType(dict(self.contrast_sections)),
        )


def _text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _only_whitespace(value: str | None) -> bool:
    return value is None or not value.strip()


def _validate_attributes(
    element: ET.Element,
    expected: set[str],
    errors: list[str],
) -> None:
    actual = set(element.attrib)
    if actual != expected:
        errors.append(
            f"{element.tag} attributes must be {sorted(expected)!r}; got {sorted(actual)!r}"
        )


def _option_from_element(element: ET.Element | None, choices: Mapping[str, str]) -> str | None:
    if element is None:
        return None
    option = element.attrib.get("option", "").strip().upper()
    return option if option in choices else None


def _empty_parse(raw: str, error: str) -> ParsedProtocol:
    return ParsedProtocol(
        document=raw,
        xml_valid=False,
        structure_valid=False,
        errors=(error,),
        sections=dict.fromkeys(ROOT_CHILDREN, ""),
        contrast_sections=dict.fromkeys(CONTRASTIVE_CHILDREN, ""),
        provisional_option=None,
        second_best_option=None,
        final_option=None,
        decision=None,
        provisional_start=raw.find("<provisional_answer") if "<provisional_answer" in raw else None,
    )


def _protocol_document(raw: str) -> str:
    match = XML_FENCE_RE.fullmatch(raw)
    return match.group("document") if match is not None else raw


def _document_errors(document: str) -> list[str]:
    errors: list[str] = []
    if document != document.strip():
        errors.append("whitespace appears before or after the XML document")
    if not document.startswith("<response>"):
        errors.append("document does not begin exactly with <response>")
    if not document.endswith("</response>"):
        errors.append("document does not end exactly with </response>")
    if "```" in document:
        errors.append("Markdown fence is only allowed as one outer ```xml wrapper")
    return errors


def _root_children(root: ET.Element, errors: list[str]) -> dict[str, ET.Element]:
    if root.tag != "response":
        errors.append(f"root tag must be response; got {root.tag}")
    _validate_attributes(root, set(), errors)
    children = list(root)
    tags = tuple(child.tag for child in children)
    if tags != ROOT_CHILDREN:
        errors.append(f"root child order/count must be {ROOT_CHILDREN!r}; got {tags!r}")
    if not _only_whitespace(root.text):
        errors.append("response may contain only whitespace outside its child elements")
    by_tag: dict[str, ET.Element] = {}
    for child in children:
        if child.tag in by_tag:
            errors.append(f"duplicate root child {child.tag}")
        by_tag[child.tag] = child
        if not _only_whitespace(child.tail):
            errors.append(f"non-whitespace text follows {child.tag}")
    return by_tag


def _contrast_children(
    contrast: ET.Element | None,
    errors: list[str],
) -> dict[str, ET.Element]:
    if contrast is None:
        return {}
    _validate_attributes(contrast, set(), errors)
    children = list(contrast)
    tags = tuple(child.tag for child in children)
    if tags != CONTRASTIVE_CHILDREN:
        errors.append(
            f"contrastive child order/count must be {CONTRASTIVE_CHILDREN!r}; got {tags!r}"
        )
    if not _only_whitespace(contrast.text):
        errors.append("contrastive_check may contain only whitespace outside children")
    by_tag: dict[str, ET.Element] = {}
    for child in children:
        if child.tag in by_tag:
            errors.append(f"duplicate contrastive child {child.tag}")
        by_tag[child.tag] = child
        if not _only_whitespace(child.tail):
            errors.append(f"non-whitespace text follows {child.tag}")
    return by_tag


def _validate_element_shapes(
    by_tag: Mapping[str, ET.Element],
    contrast_by_tag: Mapping[str, ET.Element],
    errors: list[str],
) -> None:
    for tag in ("facts", "implications"):
        element = by_tag.get(tag)
        if element is not None:
            _validate_attributes(element, set(), errors)
            if list(element):
                errors.append(f"{tag} may not contain nested XML elements")
    for tag in ("decisive_fact", "answer_changing_change"):
        element = contrast_by_tag.get(tag)
        if element is not None:
            _validate_attributes(element, set(), errors)
            if list(element):
                errors.append(f"{tag} may not contain nested XML elements")
    for tag in ("provisional_answer", "final_answer"):
        element = by_tag.get(tag)
        if element is not None:
            _validate_attributes(element, {"option"}, errors)
            if list(element):
                errors.append(f"{tag} may not contain nested XML elements")
    second = contrast_by_tag.get("second_best")
    if second is not None:
        _validate_attributes(second, {"option"}, errors)
        if list(second):
            errors.append("second_best may not contain nested XML elements")
    rereasoning = by_tag.get("rereasoning")
    if rereasoning is not None:
        _validate_attributes(rereasoning, {"decision"}, errors)
        if list(rereasoning):
            errors.append("rereasoning may not contain nested XML elements")


def _section_texts(
    by_tag: Mapping[str, ET.Element],
    contrast_by_tag: Mapping[str, ET.Element],
    errors: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    sections = {tag: _text(by_tag[tag]) if tag in by_tag else "" for tag in ROOT_CHILDREN}
    contrast_sections = {
        tag: _text(contrast_by_tag[tag]) if tag in contrast_by_tag else ""
        for tag in CONTRASTIVE_CHILDREN
    }
    for tag, text in {**sections, **contrast_sections}.items():
        if tag != "contrastive_check" and not text:
            errors.append(f"{tag} is empty")
    return sections, contrast_sections


def _protocol_options(
    by_tag: Mapping[str, ET.Element],
    contrast_by_tag: Mapping[str, ET.Element],
    choices: Mapping[str, str],
    errors: list[str],
) -> tuple[str | None, str | None, str | None, str | None]:
    provisional_element = by_tag.get("provisional_answer")
    second_element = contrast_by_tag.get("second_best")
    final_element = by_tag.get("final_answer")
    provisional = _option_from_element(provisional_element, choices)
    second = _option_from_element(second_element, choices)
    final = _option_from_element(final_element, choices)
    for tag, element, option in (
        ("provisional_answer", provisional_element, provisional),
        ("second_best", second_element, second),
        ("final_answer", final_element, final),
    ):
        if element is not None and option is None:
            errors.append(f"{tag} option is absent or invalid")
    if provisional is not None and second == provisional:
        errors.append("second_best option must differ from provisional_answer")

    rereasoning = by_tag.get("rereasoning")
    decision = None
    if rereasoning is not None:
        candidate = rereasoning.attrib.get("decision", "").strip().casefold()
        if candidate in VALID_DECISIONS:
            decision = candidate
        else:
            errors.append("rereasoning decision must be exactly retain or revise")
    if final_element is not None and final is not None:
        body = " ".join(_text(final_element).split())
        expected = " ".join(choices[final].split())
        if body != expected:
            errors.append("final_answer body must exactly match the selected answer text")
    return provisional, second, final, decision


def parse_protocol(raw: str, choices: Mapping[str, str]) -> ParsedProtocol:
    """Parse the instructed XML, allowing one outer `````xml`` fence."""
    if not raw:
        return _empty_parse(raw, "empty response")
    document = _protocol_document(raw)
    errors = _document_errors(document)
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        base = _empty_parse(raw, f"XML parse error: {exc}")
        return ParsedProtocol(
            base.document,
            False,
            False,
            tuple(errors) + base.errors,
            base.sections,
            base.contrast_sections,
            None,
            None,
            None,
            None,
            base.provisional_start,
        )
    by_tag = _root_children(root, errors)
    contrast_by_tag = _contrast_children(by_tag.get("contrastive_check"), errors)
    _validate_element_shapes(by_tag, contrast_by_tag, errors)
    sections, contrast_sections = _section_texts(by_tag, contrast_by_tag, errors)
    provisional, second, final, decision = _protocol_options(
        by_tag, contrast_by_tag, choices, errors
    )
    return ParsedProtocol(
        document=raw,
        xml_valid=True,
        structure_valid=not errors,
        errors=tuple(errors),
        sections=sections,
        contrast_sections=contrast_sections,
        provisional_option=provisional,
        second_best_option=second,
        final_option=final,
        decision=decision,
        provisional_start=raw.find("<provisional_answer") if "<provisional_answer" in raw else None,
    )


def _sentence_context(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), text.rfind(";", 0, start)) + 1
    positions = [
        position
        for position in (text.find(".", end), text.find("\n", end), text.find(";", end))
        if position >= 0
    ]
    right = min(positions) if positions else len(text)
    return text[left:right]


def _commitments(text: str, choices: Mapping[str, str]) -> list[CommitmentEvent]:
    events: list[CommitmentEvent] = []
    for pattern in (EXPLICIT_LABEL_COMMITMENT_RE, BARE_LABEL_COMMITMENT_RE):
        for match in pattern.finditer(text):
            label = match.group(1).upper()
            context = _sentence_context(text, match.start(), match.end())
            if label in choices and not COMMITMENT_EXCLUSION.search(context):
                events.append({"option": label, "start": match.start(), "evidence": match.group(0)})
    normalized_choices = {label: normalize_text(value) for label, value in choices.items()}
    for match in TEXT_COMMITMENT_RE.finditer(text):
        context = _sentence_context(text, match.start(), match.end())
        if COMMITMENT_EXCLUSION.search(context):
            continue
        candidate = normalize_text(match.group(1))
        hits = [
            label
            for label, choice in normalized_choices.items()
            if choice and (candidate == choice or candidate.startswith(choice + " "))
        ]
        if len(hits) == 1:
            events.append({"option": hits[0], "start": match.start(), "evidence": match.group(0)})
    events.sort(key=lambda event: (int(event["start"]), str(event["option"])))
    return [event for index, event in enumerate(events) if not index or event != events[index - 1]]


def extract_protocol_final_answer(raw: str, choices: Mapping[str, str]) -> ExtractedAnswer:
    """Extract the instructed final answer without requiring the whole XML document to parse."""
    matches = list(FINAL_TAG_RE.finditer(raw))
    if len(matches) != 1:
        return ExtractedAnswer(
            None,
            "missing_final_answer" if not matches else "duplicate_final_answer",
            None,
            (),
        )
    match = matches[0]
    attr = OPTION_ATTR_RE.search(match.group("attrs") or "")
    option = attr.group(1).upper() if attr else None
    body = " ".join(re.sub(r"<[^>]+>", " ", match.group("body")).split())
    if option is None or option not in choices:
        return ExtractedAnswer(None, "invalid_label", match.group(0), (option,) if option else ())
    if not body:
        return ExtractedAnswer(None, "empty_final_answer", match.group(0), (option,))
    if " ".join(body.split()) != " ".join(choices[option].split()):
        body_answer = extract_answer(body, choices)
        if body_answer.option is not None and body_answer.option != option:
            return ExtractedAnswer(
                None,
                "conflicting",
                match.group(0),
                tuple(sorted({option, body_answer.option})),
            )
        return ExtractedAnswer(None, "body_mismatch", match.group(0), (option,))
    return ExtractedAnswer(option, "parsed", match.group(0), (option,))


def _token_count(text: str) -> int:
    return len(content_tokens(text))


def _informative(text: str, minimum_tokens: int) -> bool:
    """Reject empty, extremely short, and mechanically repetitive section filler."""
    tokens = content_tokens(text)
    if len(tokens) < minimum_tokens:
        return False
    unique = set(tokens)
    if len(unique) < min(4, minimum_tokens):
        return False
    most_common = max(tokens.count(token) for token in unique)
    return most_common / len(tokens) <= 0.5


def _mentions_option(text: str, option: str | None, choices: Mapping[str, str]) -> bool:
    if option is None:
        return False
    if re.search(rf"\boption\s+{re.escape(option)}\b", text, re.IGNORECASE):
        return True
    if re.search(rf"\b{re.escape(option)}\b", text):
        return True
    choice = normalize_text(choices[option])
    normalized = normalize_text(text)
    return bool(choice and (choice in normalized or normalized in choice))


def _failure_judgment(instructed: bool, status: str) -> Judgment:
    adverse = 1.0 if instructed else None
    zero = 0.0 if instructed else None
    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": False,
        "generation_status": status,
        "minimum_repair_scaffold": zero,
        "contrastive_scaffold_complete": zero,
        "facts_anchor_recall": zero,
        "implications_anchor_recall": zero,
        "preanswer_anchor_recall": zero,
        "shared_anchor_recall": zero,
        "firm_preprovisional_commitment": adverse,
        "xml_document_valid": False if instructed else None,
        "protocol_complete": False if instructed else None,
        "accuracy": 0,
        "accuracy_parseable": False,
        "final_option": None,
        "answer_extraction_status": "generation_failure",
        "answer_candidates": [],
        "parser_errors": [f"generation status: {status}"],
        "repair_readiness_criteria": {},
        "section_substantive": {},
        "section_token_counts": {},
        "provisional_option": None,
        "second_best_option": None,
        "rereasoning_decision": None,
        "decision_consistent": False,
        "early_commitments": [],
    }


def _baseline_judgment(question: Question, status: str, raw: str) -> Judgment:
    answer = extract_answer(raw, question.choices)
    parseable = answer.status == "parsed" and answer.option is not None
    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": True,
        "generation_status": status,
        "minimum_repair_scaffold": None,
        "contrastive_scaffold_complete": None,
        "facts_anchor_recall": None,
        "implications_anchor_recall": None,
        "preanswer_anchor_recall": None,
        "shared_anchor_recall": None,
        "firm_preprovisional_commitment": None,
        "xml_document_valid": None,
        "protocol_complete": None,
        "accuracy": int(bool(parseable and answer.option == question.gold)),
        "accuracy_parseable": parseable,
        "final_option": answer.option,
        "answer_extraction_status": answer.status,
        "answer_candidates": list(answer.candidates),
        "parser_errors": [],
        "repair_readiness_criteria": {},
        "section_substantive": {},
        "section_token_counts": {},
        "provisional_option": None,
        "second_best_option": None,
        "rereasoning_decision": None,
        "decision_consistent": False,
        "early_commitments": [],
    }


def _section_counts(parsed: ParsedProtocol) -> dict[str, int]:
    sections = parsed.sections
    contrast = parsed.contrast_sections
    return {
        "facts": _token_count(sections["facts"]),
        "implications": _token_count(sections["implications"]),
        "provisional_answer": _token_count(sections["provisional_answer"]),
        "second_best": _token_count(contrast["second_best"]),
        "decisive_fact": _token_count(contrast["decisive_fact"]),
        "answer_changing_change": _token_count(contrast["answer_changing_change"]),
        "rereasoning": _token_count(sections["rereasoning"]),
        "final_answer": _token_count(sections["final_answer"]),
    }


def _decision_consistent(parsed: ParsedProtocol, choices: Mapping[str, str]) -> bool:
    return bool(
        parsed.decision in VALID_DECISIONS
        and parsed.provisional_option in choices
        and parsed.final_option in choices
        and (
            (parsed.decision == "retain" and parsed.final_option == parsed.provisional_option)
            or (parsed.decision == "revise" and parsed.final_option != parsed.provisional_option)
        )
    )


def _substantive_sections(
    question: Question,
    parsed: ParsedProtocol,
    counts: Mapping[str, int],
    lexical: Mapping[str, float],
    decisive_overlap: float,
    decision_consistent: bool,
) -> dict[str, bool]:
    sections = parsed.sections
    contrast = parsed.contrast_sections
    return {
        "facts": bool(_informative(sections["facts"], 4) and lexical["facts_anchor_recall"] > 0),
        "implications": bool(
            _informative(sections["implications"], 6) and lexical["implications_anchor_recall"] > 0
        ),
        "provisional_answer": bool(
            parsed.provisional_option in question.choices
            and _informative(sections["provisional_answer"], 4)
            and _mentions_option(
                sections["provisional_answer"], parsed.provisional_option, question.choices
            )
        ),
        "second_best": bool(
            parsed.second_best_option in question.choices
            and parsed.second_best_option != parsed.provisional_option
            and _informative(contrast["second_best"], 5)
            and _mentions_option(
                contrast["second_best"], parsed.second_best_option, question.choices
            )
        ),
        "decisive_fact": bool(_informative(contrast["decisive_fact"], 3) and decisive_overlap > 0),
        "answer_changing_change": bool(
            _informative(contrast["answer_changing_change"], 5)
            and COUNTERFACTUAL_CUES.search(contrast["answer_changing_change"])
            and _mentions_option(
                contrast["answer_changing_change"],
                parsed.second_best_option,
                question.choices,
            )
        ),
        "rereasoning": bool(
            _informative(sections["rereasoning"], 5)
            and decision_consistent
            and (
                _mentions_option(
                    sections["rereasoning"], parsed.provisional_option, question.choices
                )
                or _mentions_option(sections["rereasoning"], parsed.final_option, question.choices)
            )
        ),
        "final_answer": bool(
            parsed.final_option in question.choices
            and " ".join(sections["final_answer"].split())
            == " ".join(question.choices.get(parsed.final_option or "", "").split())
        ),
    }


def _protocol_complete(parsed: ParsedProtocol, choices: Mapping[str, str]) -> bool:
    return bool(
        parsed.structure_valid
        and parsed.provisional_option in choices
        and parsed.second_best_option in choices
        and parsed.second_best_option != parsed.provisional_option
        and parsed.final_option in choices
        and parsed.decision in VALID_DECISIONS
    )


def _instructed_judgment(
    question: Question,
    status: str,
    raw: str,
    lexical_reference: Mapping[str, object],
) -> Judgment:
    parsed = parse_protocol(raw, question.choices)
    sections = parsed.sections
    contrast = parsed.contrast_sections
    lexical = score_preanswer(
        question.stem,
        sections["facts"],
        sections["implications"],
        lexical_reference,
    )
    preprovisional = (
        raw[: parsed.provisional_start] if parsed.provisional_start is not None else raw
    )
    commitments = _commitments(preprovisional, question.choices)
    counts = _section_counts(parsed)
    decisive_overlap = score_preanswer(
        question.stem,
        contrast["decisive_fact"],
        "",
        lexical_reference,
    )["preanswer_anchor_recall"]
    decision_consistent = _decision_consistent(parsed, question.choices)
    substantive = _substantive_sections(
        question,
        parsed,
        counts,
        lexical,
        decisive_overlap,
        decision_consistent,
    )
    protocol_complete = _protocol_complete(parsed, question.choices)
    contrastive_complete = bool(
        protocol_complete
        and substantive["second_best"]
        and substantive["decisive_fact"]
        and substantive["answer_changing_change"]
        and substantive["rereasoning"]
        and decision_consistent
    )
    minimum_scaffold = bool(
        protocol_complete and all(substantive.values()) and decision_consistent and not commitments
    )
    answer = extract_protocol_final_answer(raw, question.choices)
    parseable = answer.status == "parsed" and answer.option is not None
    return {
        "judge_version": JUDGE_VERSION,
        "generation_usable": True,
        "generation_status": status,
        "minimum_repair_scaffold": float(minimum_scaffold),
        "contrastive_scaffold_complete": float(contrastive_complete),
        "repair_readiness_criteria": {
            "exact_protocol": protocol_complete,
            "all_sections_substantive": all(substantive.values()),
            "contrastive_check": contrastive_complete,
            "decision_and_final_consistency": bool(
                decision_consistent and substantive["final_answer"]
            ),
        },
        "firm_preprovisional_commitment": float(bool(commitments)),
        "xml_document_valid": parsed.xml_valid,
        "protocol_complete": protocol_complete,
        "section_substantive": substantive,
        "section_token_counts": counts,
        "provisional_option": parsed.provisional_option,
        "second_best_option": parsed.second_best_option,
        "rereasoning_decision": parsed.decision,
        "decision_consistent": decision_consistent,
        "final_option": answer.option,
        "accuracy": int(bool(parseable and answer.option == question.gold)),
        "accuracy_parseable": parseable,
        "answer_extraction_status": answer.status,
        "answer_candidates": list(answer.candidates),
        "parser_errors": list(parsed.errors),
        "early_commitments": commitments,
        "facts_anchor_recall": lexical["facts_anchor_recall"],
        "implications_anchor_recall": lexical["implications_anchor_recall"],
        "preanswer_anchor_recall": lexical["preanswer_anchor_recall"],
        "shared_anchor_recall": lexical["shared_anchor_recall"],
    }


def judge(
    question: Question,
    status: str,
    raw: str | None,
    condition_id: str,
    lexical_reference: Mapping[str, object],
) -> Judgment:
    """Compute versioned deterministic measurements for one cell."""
    instructed = condition_id != "zero"
    if status != CellStatus.COMPLETED.value or raw is None or not raw.strip():
        return _failure_judgment(instructed, status)
    if not instructed:
        return _baseline_judgment(question, status, raw)
    return _instructed_judgment(question, status, raw, lexical_reference)
