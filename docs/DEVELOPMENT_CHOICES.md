# Development choices and parameter selection

This appendix documents the actual experimental/configuration parameters used during development and the criterion for each final choice. It is written to support AAAI reproducibility-checklist item **4.2 = yes**. Singleton settings are explicitly identified as **no sweep** rather than retroactively inventing a range.

| Parameter / design choice | Number/range of values tried or evaluated during development | Final value | Selection criterion |
|---|---|---|---|
| Temperature | 1 value (**no sweep**) | `0` | Minimize sampling variability and make repeated inference as deterministic as the hosted route permits; not selected by outcome. |
| Per-cell generation seed | 1 deterministic rule (**no seed sweep**) | `int(cell_id[:8], 16)` | Stable identity-derived seed for each scheduled cell; not selected by outcome. |
| `top_p` | 0 explicit overrides (**no sweep**) | provider default | Not explicitly overridden. |
| `top_k` | 0 explicit overrides (**no sweep**) | provider default | Not explicitly overridden. |
| Other sampling controls | 0 explicit overrides beyond temperature/seed | provider defaults | Avoid introducing additional tuned sampling parameters. |
| Procedural instruction wording in the reported run | 1 frozen eight-stage protocol wording (**no wording sweep in the reported run**) | protocol SHA-256 `86b92dcf90caef0528ec7270c28bcd50f65e2a9bbc187ea8348824f06001b4c1` | Freeze one wording so the experiment varies duplication/placement rather than instruction wording. |
| Placement/copy conditions | all 8 values of the 2×2×2 system/before/after design | `zero`, `system`, `before`, `after`, `system_before`, `system_after`, `before_after`, `system_before_after` | Exhaustive evaluation of all combinations; no winning placement was selected. |
| Questions per dataset | 1 final study size (**no outcome-based sweep**) | 100 per dataset (300 total) | Fixed study scope/cost and balanced dataset contribution. |
| Model panel | 1 seven-model panel (**no outcome-based model selection**) | the seven IDs listed in `FINAL_EXPERIMENT_CONFIGURATION.md` | Span multiple instruction-tuned model families/sizes/routes; models were not retained/dropped according to duplication effect. |
| Output ceilings | model-specific smoke-calibrated ceilings; one documented revision for Mistral Large | 1,280; 1,280; 1,280; 4,352; 2,304; 2,304; 2,816 tokens | Avoid conditional truncation/resampling while keeping bounded requests. Mistral Large was revised **1,792→2,304** because a current-protocol smoke included an ordinary stop-terminated 1,663-token completion (p99 1,504.35), making 1,792 too close to the observed tail. This engineering revision was not chosen by paper endpoints. |
| Provider/backend route candidates | candidate backend/provider lists in frozen `paper-run/config/models.json` and backend order in `paper-run/config/preflight.json` | exact routes in `paper-run/config/routes.json` | Functional probes plus concurrency-capacity preflight, never answer accuracy/compliance or duplication effect. |
| Route-qualification concurrency | 1 value (**no sweep for scientific selection**) | 8 concurrent requests | Verify that the candidate route could sustain the intended concurrent workload. |
| Final per-route concurrency cap | 1 value after preflight (**no scientific sweep**) | 16 | Throughput/orchestration setting after route qualification; not selected by scientific outcome. |
| Bootstrap resamples | 1 value (**no sweep**) | 10,000 | Fixed Monte Carlo precision for confidence intervals. |
| Sign-flip draws | 1 value (**no sweep**) | 50,000 | Fixed Monte Carlo precision for paired randomization tests. |
| Confidence level | 1 value (**no sweep**) | 95% | Conventional inferential reporting level fixed for the analysis. |
| Multiple-testing correction | 1 method (**no sweep**) | Holm | Family-wise error control without choosing a correction by significance outcome. |
| Human-audit null confirmation rate `p0` | 1 pre-specified value | 0.80 | Design threshold for minimum acceptable machine/human directional confirmation. |
| Human-audit alternative `p1` | 1 pre-specified value | 0.95 | Design alternative used for prospective sample-size calculation. |
| Human-audit alpha | 1 pre-specified value | 0.05 | One-sided exact-binomial type-I error target. |
| Human-audit target power | 1 pre-specified value | 0.80 | Prospective power target; implied primary `n=30`, critical confirmations `28`, achieved power 0.812 at `p1=.95`. |
| Human-audit sentinel fraction | 1 pre-specified value | 0.10 per sentinel type relative to primary n | Small descriptive tie/degradation controls; not separately powered. |
| Human-audit sampling/side assignment | 1 stable-hash deterministic procedure | seed `instruction-duplication-aaai27-effect-validation-2026-08-19-v1` | Reproducible sampling and A/B orientation without using gold answer, response quality, score magnitude, or treatment outcome beyond the predeclared machine-change stratum. |

## Frozen protocol wording

The reported generation used exactly this instruction text:

> Use the following eight headings, in order, to answer the question. Complete every section. Do not select an answer before the Provisional answer section.
>
> 1. Facts — Discuss every important fact in the question, including relevant timing, laterality, negation, measurements, and qualifiers.
> 2. Implications — Explain what the facts support or argue against and how they distinguish the answer choices.
> 3. Provisional answer — Select the best answer and explain why.
> 4. Best alternative — Select a different answer as the best alternative and explain why it is less suitable.
> 5. Decisive distinction — Identify the fact that best distinguishes the provisional answer from the best alternative.
> 6. What would change the answer — Describe the smallest change to the question that would make the best alternative the best answer, and explain why.
> 7. Reconsideration — Reconsider the provisional answer using the original facts. State whether you retain or revise it, and explain why.
> 8. Final answer — State the selected option and its answer text.

## Deterministic-judge refinement is measurement-method development, not hyperparameter tuning

After generation was frozen, the deterministic judge was refined to better operationalize the paper's stated measurement constructs and checked against held-out/manual examples. This was **measurement-method development, not a numerical hyperparameter sweep**. No grid/random/Bayesian search over judge parameters was conducted, and judge revisions were **not selected by maximizing the duplication effect**, statistical significance, or any treatment/control separation. The frozen generations were not regenerated during that process. The final judge and its validation artifacts are included so that this measurement-development history is auditable without mischaracterizing it as outcome-optimized hyperparameter tuning.
