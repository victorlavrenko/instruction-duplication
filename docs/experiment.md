# Experiment design

## Research question

The experiment tests whether repeating the same reasoning instructions changes how completely
an instruction-tuned model exposes a requested reasoning trajectory. Final-answer accuracy is a
separate outcome. The motivating engineering question is whether duplication sometimes produces
more explicit facts, implications, alternatives, counterfactuals, and reconsideration that a
downstream auditing or repair system could inspect.

The study does not claim that visible reasoning is faithful to hidden model computation or that a
complete trajectory is medically correct.

## Factorial conditions

Every selected question is evaluated under the same model settings in eight conditions. The
conditions form a full factorial design over three possible locations for an exact copy of the
reasoning protocol:

| Condition | System message | Before question | After question |
|---|:---:|:---:|:---:|
| `zero` |  |  |  |
| `system` | ✓ |  |  |
| `before` |  | ✓ |  |
| `after` |  |  | ✓ |
| `system_before` | ✓ | ✓ |  |
| `system_after` | ✓ |  | ✓ |
| `before_after` |  | ✓ | ✓ |
| `system_before_after` | ✓ | ✓ | ✓ |

The zero-copy condition receives only a neutral request to reason through the multiple-choice
question and give a final answer. The main duplication estimand is the paired mean of the three
two-copy conditions minus the paired mean of the three one-copy conditions. One-copy placement,
trailing-copy, three-copy, and full 2×2×2 factorial effects are reported separately.

## Labeled-section reasoning protocol

The protocol requests eight labeled roles, in this order:

1. Facts
2. Implications
3. Provisional answer
4. Best alternative
5. Decisive distinction
6. What would change the answer
7. Reconsideration
8. Final answer

The prompt simply asks the model to use those eight section headings in order and complete every
section. It says nothing about XML, Markdown, numbering, punctuation, or alternative presentation
formats. The multiple-choice question is rendered as ordinary text.

The complete authoritative wording is defined once in `src/instruction_duplication/protocol.py`.
The protocol and condition table are hashed into the experiment manifest. Any prompt change
therefore requires a fresh generation workspace.

### Prompt-design rationale

The prompt specifies only what the experiment measures: eight distinct reasoning roles, their
order, and the prohibition on selecting an answer before the provisional-answer role. The role
descriptions define the work each section must perform. It contains no serialization rules,
format menu, example response, or restatement of the measurement criteria. Exact copies of this
same text are used in every instructed location.

## Format-tolerant role recognition

The deterministic parser accepts the canonical titles plus a small frozen alias set such as
`Case facts`, `Initial answer`, `Second-best answer`, `Key distinction`, `What would change the
answer`, and `Re-evaluation`. A role boundary may be a plain, numbered, Markdown, or voluntarily
emitted XML-like heading. A lone reversed tag can identify a role only when no ordinary boundary
for that role exists. Conventional XML closing tags are ignored when a corresponding opening
boundary exists, so closing tags do not create false duplicates.

Each role begins at its recognized heading and ends at the next recognized role heading. A heading
and its content may share one line when separated by a colon or dash. Presentation markup is
stripped from the recovered content. Exact markup never changes primary role-completeness credit.

## Core measurement outcomes

Judge v2 keeps the four main constructs separate. No cross-component AE score is constructed.

1. `required_section_count` (0–8): how many of the eight requested reasoning sections have a recognizable semantic boundary. This is deliberately a structural measure and is tolerant of plain text, numbering, Markdown, emphasis, aliases, and voluntarily emitted XML-like headings.
2. `nontrivial_section_count` (0–8): how many recognized sections contain a non-trivial body. This is deliberately separate from whether the body performs the requested reasoning role correctly.
3. `preprovisional_tfidf_recall` (0–1): PubMed-IDF-weighted lexical recall of specific material from the question stem that appears in recovered Facts or Implications before the provisional-answer boundary. The IDF source is the frozen global PubMed yearly-count table, not frequencies estimated from the experiment sample. This measures visible stem exposure, not medical correctness and not precision against irrelevant extra prose.
4. `contrastive_discussion_score` (0–1), with `contrastive_discussion_count` (0–5): whether the trajectory visibly performs five substantive stages — (a) selects a provisional option with rationale, (b) selects a different best alternative with rationale, (c) discusses a decisive distinction, (d) gives a case-specific counterfactual that would make that declared alternative win, and (e) reconsiders the provisional answer with an explicit retain/revise decision grounded in the case.

Accuracy, premature commitment, the older role-by-role substantive diagnostics, uniqueness/order checks, generation success, and lexical diagnostics remain separate outcomes. For a future preregistered run, `nontrivial_section_count` is the primary compliance endpoint; the other three core measures and accuracy are key secondary endpoints. The 2026-08-12 run was used to develop and manually audit judge v2 and therefore must be treated as judge-development/exploratory evidence rather than a fresh confirmatory test.

## Format-neutral parsing contract

Role recognition operates on presentation-normalized text. Lightweight Markdown emphasis does not change section presence or option recovery; an emphasized inline phrase such as `**Best alternative: C. ...**` is treated as body text rather than a second section boundary. Immediate body labels such as `Provisional answer: C` do not create duplicate headings. Explicit option labels are preferred over fuzzy answer-text matching, while normalized choice-text matching handles known benchmark punctuation/export debris. Ambiguous prose that never actually commits to an option is conservatively scored as not discussed instead of being guessed.

## Supporting semantic-role diagnostics

The older `substantive_role_count` and per-role completion flags remain useful diagnostics. Their role-function rules are:

The judge applies frozen mechanical rules to the recovered body of each role:

- Facts must contain non-trivial content anchored to the stem and must not become an explicit
  multi-option answer list.
- Implications must contain a case-grounded explanation rather than generic filler.
- Provisional answer must identify an option and provide a rationale beyond the option name.
- Best alternative must identify a different option and explain why it loses in the actual case.
- Decisive distinction must contain a case-grounded discriminator between the provisional answer
  and best alternative.
- What would change the answer must propose a concrete case-fact change and explicitly make the stated
  best alternative the winner. Merely changing the task, reversing the question, or making the
  alternative “more relevant” does not pass.
- Reconsideration must say `retain` or `revise`, remain consistent with the provisional and final
  selections, and reconnect to case-specific reasoning.
- Final answer must identify one available option by label or unambiguous answer text.

These are content and role-function checks, not medical truth judgments. A blinded matched-pair
export is produced for human validation.

## Lexical exposure and fact diagnostics

The principal lexical reference uses document frequencies from a pinned table covering
15,103,887 PubMed abstracts from 2010–2024. For a stem term `t`:

```text
idf(t) = log((N + 1) / (df(t) + 1)) + 1
stem_weight(t) = (1 + log(tf_stem(t))) × idf(t)
```

Candidate term frequency is capped at the term frequency available in the stem, so repeating one
term in a response cannot create unlimited credit. The main lexical endpoints are pre-answer
weighted recall and weighted material shared between Facts and Implications. Supporting outputs
include high-IDF recall, content-token count, credited IDF mass per 100 content tokens,
polarity-aware anchor recall, an automatic atomic-fact inventory, and hard-qualifier preservation.

Lexical exposure does not establish proposition truth, medical validity, or faithful hidden
reasoning. The package intentionally constructs no cross-component repair-readiness or AE score.

## Accuracy and generation failures

Accuracy is parsed from the Final answer role independently of trajectory quality. The option may
be written as a label or as unambiguous answer text. If a voluntarily emitted option attribute and
the visible body deterministically identify different choices, the answer is conflicting and
unparseable.

Generation failures are retained in intention-to-treat analyses: beneficial instructed outcomes
receive zero, the adverse premature-commitment/failure endpoint receives one, and accuracy receives
zero. A response terminated because it reaches the fixed output ceiling remains terminal rather
than being conditionally resampled with a larger ceiling.

## Statistical analysis

All primary contrasts are paired within model-question blocks and clustered equally by question.
Confidence intervals use question-clustered bootstrap resampling; p-values use question-clustered
sign-flip tests. The primary endpoint is unadjusted because it is the single declared confirmatory
test. The five key secondaries use Holm correction. Model-specific duplication effects use Holm
correction across models separately within each endpoint.

Placement, dataset, question-complexity, response-length, and factorial decompositions are
secondary or exploratory and are labeled as such. Positive examples are illustrative and cannot
estimate prevalence.

## New workspaces and independent questions

Version 3.0 is not generation-compatible with 2.x because the prompt changed. Regenerate into a
fresh workspace. Older workspaces can still be supplied with repeatable `--exclude-workspace`
arguments; their immutable question ledgers are used only to prevent question reuse.
