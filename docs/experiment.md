# Experiment design

## Research question

The experiment tests whether repeating a detailed response protocol, and changing where it appears in the prompt, changes instruction following and answer quality.

Each selected question is evaluated under the same model settings in eight conditions. The conditions form a full factorial design over three possible protocol locations:

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

The `zero` condition receives a neutral request to reason through the question and provide a final answer. The other conditions receive one to three exact copies of the protocol.

## Protocol

The protocol requires one XML document with these sections in order:

1. facts from the stem;
2. implications of those facts;
3. a provisional answer;
4. a contrastive check against the second-best option;
5. rereasoning that retains or revises the provisional answer;
6. the final answer and exact option text.

The complete text is defined once in `src/instruction_duplication/protocol.py`. Its content and the condition table are hashed into the experiment manifest.

## Questions and models

The default run draws an equal number of questions from three pinned medical multiple-choice datasets:

- MedQA;
- MedXpertQA;
- AfriMedQA.

Rows are normalized to a common question schema. Ambiguous labels, missing stems, duplicate choices, unresolved media references, and other unsafe-to-guess cases are rejected and recorded in `dataset-audit.json`.

The default model panel is defined in `models.py`. Provider identifiers, preferred routes, prices, concurrency limits, and request ceilings are part of the saved model configuration.

## Measurements

Answer accuracy and protocol compliance are separate outcomes.

Protocol scoring is local and deterministic. A response receives structural credit only when it contains the expected XML elements, order, attributes, and non-empty content. A single outer ` ```xml ` fence is tolerated; other wrappers or malformed XML are not repaired.

Additional lexical measures estimate whether the response covers important details from the question stem, including polarity, laterality, timing, and quantities. These are anchor-recall diagnostics, not semantic similarity scores.

## Analysis

The primary analysis is paired by question and uses all planned cells. Failed, refused, truncated, or missing generations are assigned the declared adverse or zero outcomes for intention-to-treat analysis. Complete successful responses are also reported as a sensitivity analysis.

Confidence intervals and permutation tests are clustered at the question level. The pooled endpoints represent distinct measurement layers and are reported individually without a joint cross-endpoint multiplicity correction; several unadjusted pooled p-values should therefore still be interpreted with caution. Model-specific tests use Holm correction across models separately within each endpoint. Trailing-copy placement contrasts, dataset-specific heterogeneity, and exploratory diagnostics are unadjusted and explicitly exploratory or descriptive.

The authoritative machine-readable output is `analysis.json`; `report.txt` is a rendering of the same results.
