from __future__ import annotations

from instruction_duplication.judge import extract_protocol_final_answer, judge, parse_protocol
from instruction_duplication.provider import fake_response
from instruction_duplication.records import GenerationCell


def valid_xml(question):
    cell = GenerationCell(
        cell_id="a" * 64,
        question_id=question.id,
        model_id="test-model",
        condition_id="system",
        copies=1,
        dataset=question.dataset,
        stem=question.stem,
        choices=question.choices,
        gold=question.gold,
    )
    return fake_response(cell)["choices"][0]["message"]["content"]


def test_valid_protocol_gets_strict_scaffold(question, lexical_reference):
    result = judge(question, "completed", valid_xml(question), "system", lexical_reference)
    assert result["xml_document_valid"] is True
    assert result["protocol_complete"] is True
    assert result["minimum_repair_scaffold"] == 1.0
    assert result["accuracy"] == 1


def test_single_xml_markdown_fence_is_accepted(question, lexical_reference):
    raw = "```xml\n" + valid_xml(question) + "\n```"
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["xml_document_valid"] is True
    assert result["protocol_complete"] is True
    assert result["minimum_repair_scaffold"] == 1.0
    assert result["parser_errors"] == []


def test_invalid_xml_can_still_have_a_parseable_protocol_final_answer(question, lexical_reference):
    raw = valid_xml(question).replace("<facts>", "<facts>R&D evidence: ", 1)
    parsed = parse_protocol(raw, question.choices)
    answer = extract_protocol_final_answer(raw, question.choices)
    result = judge(question, "completed", raw, "system", lexical_reference)

    assert parsed.xml_valid is False
    assert answer.status == "parsed"
    assert answer.option == question.gold
    assert result["xml_document_valid"] is False
    assert result["protocol_complete"] is False
    assert result["accuracy_parseable"] is True
    assert result["accuracy"] == 1


def test_non_xml_markdown_fence_is_rejected(question, lexical_reference):
    raw = "```\n" + valid_xml(question) + "\n```"
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["protocol_complete"] is False
    assert result["minimum_repair_scaffold"] == 0.0


def test_unclosed_duplicate_and_misplaced_elements_fail(question, lexical_reference):
    raw = valid_xml(question)
    variants = [
        raw.replace("</facts>", "", 1),
        raw.replace("</facts>", "</facts><facts>duplicate content</facts>", 1),
        raw.replace("<contrastive_check>", "").replace("</contrastive_check>", ""),
    ]
    for value in variants:
        result = judge(question, "completed", value, "system", lexical_reference)
        assert result["protocol_complete"] is False
        assert result["minimum_repair_scaffold"] == 0.0


def test_empty_final_body_fails_scaffold_and_accuracy(question, lexical_reference):
    raw = valid_xml(question).replace(f">{question.gold_text}</final_answer>", "></final_answer>")
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["protocol_complete"] is False
    assert result["minimum_repair_scaffold"] == 0.0
    assert result["accuracy"] == 0
    assert result["answer_extraction_status"] == "empty_final_answer"


def test_literal_decision_placeholder_is_invalid(question):
    raw = valid_xml(question).replace('decision="retain"', 'decision="retain|revise"')
    parsed = parse_protocol(raw, question.choices)
    assert parsed.structure_valid is False
    assert any("decision" in error for error in parsed.errors)


def test_early_commitment_is_detected_and_blocks_minimum_scaffold(question, lexical_reference):
    raw = valid_xml(question).replace("<facts>", "<facts>The answer is option A. ", 1)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["firm_preprovisional_commitment"] == 1.0
    assert result["minimum_repair_scaffold"] == 0.0
    assert result["early_commitments"]


def test_generation_failure_has_itt_values(question, lexical_reference):
    result = judge(question, "truncated", None, "system", lexical_reference)
    assert result["generation_usable"] is False
    assert result["minimum_repair_scaffold"] == 0.0
    assert result["firm_preprovisional_commitment"] == 1.0
    assert result["accuracy"] == 0


def test_baseline_keeps_repair_metrics_undefined(question, lexical_reference):
    result = judge(question, "completed", "Final answer: A", "zero", lexical_reference)
    assert result["accuracy"] == 1
    assert result["minimum_repair_scaffold"] is None


def test_final_answer_text_is_character_exact_after_whitespace(question, lexical_reference):
    target = f'<final_answer option="{question.gold}">{question.gold_text}</final_answer>'
    replacement = f'<final_answer option="{question.gold}">{question.gold_text}!!!</final_answer>'
    raw = valid_xml(question).replace(target, replacement)
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["protocol_complete"] is False
    assert any("exactly match" in error for error in result["parser_errors"])


def test_lowercase_article_is_not_early_option_commitment(question, lexical_reference):
    raw = valid_xml(question).replace(
        "<facts>", "<facts>The answer is a treatment category under review. ", 1
    )
    result = judge(question, "completed", raw, "system", lexical_reference)
    assert result["firm_preprovisional_commitment"] == 0.0
