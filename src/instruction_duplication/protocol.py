"""Experimental conditions and the single authoritative protocol text."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import TypedDict

from .io_utils import sha256_json
from .json_types import JsonObject


class ChatMessage(TypedDict):
    """One OpenAI-compatible chat message."""

    role: str
    content: str


PROTOCOL = """Your entire response must be exactly one XML document. The first characters must be <response>, and nothing may appear before or after that document. Do not use Markdown fences. XML-escape &, <, and > when they occur in text. Do not state, guess, or imply a preferred answer before <provisional_answer>.

Use every element below exactly once and in this order. Replace X and Y with option labels; Y must differ from X. Replace the rereasoning decision with exactly retain or revise. Keep every element substantive and non-empty. The final_answer body must be exactly the selected answer text.

<response>
  <facts>Discuss every important fact stated in the stem, preserving timing, laterality, negation, measurements, and other qualifiers. Do not select an answer.</facts>
  <implications>Explain what those facts support or argue against and which facts distinguish the choices. Do not select an answer.</implications>
  <provisional_answer option="X">State one provisional answer and why it currently appears best.</provisional_answer>
  <contrastive_check>
    <second_best option="Y">State the second-best answer and why it loses under the actual stem.</second_best>
    <decisive_fact>State the actual stem fact that most strongly separates X from Y.</decisive_fact>
    <answer_changing_change>State the smallest substantive stem change that would make Y the best answer, and why.</answer_changing_change>
  </contrastive_check>
  <rereasoning decision="retain">Check the provisional answer against every important fact and the contrastive change. Use retain if the final answer is unchanged and revise if it changes; identify the decisive fact.</rereasoning>
  <final_answer option="X">Exact answer text</final_answer>
</response>"""

NEUTRAL_SYSTEM = (
    "Answer the multiple-choice question accurately. Follow any output protocol that is provided."
)
BASELINE_REQUEST = "Reason through the question and give your final answer."


@dataclass(frozen=True, slots=True)
class Condition:
    """One factorial instruction-placement condition."""

    id: str
    system: bool
    before: bool
    after: bool

    @property
    def copies(self) -> int:
        """Return the number of protocol copies in this condition."""
        return int(self.system) + int(self.before) + int(self.after)

    def to_dict(self) -> JsonObject:
        """Serialize the condition."""
        return {
            "id": self.id,
            "system": self.system,
            "before": self.before,
            "after": self.after,
        }


CONDITIONS: tuple[Condition, ...] = (
    Condition("zero", False, False, False),
    Condition("system", True, False, False),
    Condition("before", False, True, False),
    Condition("after", False, False, True),
    Condition("system_before", True, True, False),
    Condition("system_after", True, False, True),
    Condition("before_after", False, True, True),
    Condition("system_before_after", True, True, True),
)
CONDITION_BY_ID = {condition.id: condition for condition in CONDITIONS}
PROTOCOL_HASH = sha256_json({"protocol": PROTOCOL, "conditions": [c.to_dict() for c in CONDITIONS]})


def format_question(stem: str, choices: Mapping[str, str]) -> str:
    """Render a multiple-choice question in a deterministic XML wrapper."""
    lines = ["<question>", "  <stem>", escape(stem.strip()), "  </stem>", "  <choices>"]
    for label, text in choices.items():
        lines.append(f'    <choice option="{escape(str(label))}">{escape(str(text))}</choice>')
    lines.extend(["  </choices>", "</question>"])
    return "\n".join(lines)


def render_messages(
    stem: str,
    choices: Mapping[str, str],
    condition_id: str,
) -> list[ChatMessage]:
    """Render chat messages for one experiment cell."""
    try:
        condition = CONDITION_BY_ID[condition_id]
    except KeyError as exc:
        raise ValueError(f"unknown condition: {condition_id}") from exc

    system = NEUTRAL_SYSTEM + ("\n\n" + PROTOCOL if condition.system else "")
    user_parts: list[str] = []
    if condition.before:
        user_parts.append(PROTOCOL)
    user_parts.append(format_question(stem, choices))
    if condition.after:
        user_parts.append(PROTOCOL)
    if condition.copies == 0:
        user_parts.append(BASELINE_REQUEST)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
