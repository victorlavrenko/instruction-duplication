# Final experiment configuration

This document supports AAAI reproducibility-checklist item **4.13 = yes** and is descriptive only; it does not alter the frozen run.

This appendix lists the final settings used by every reported model/algorithm. It is generated from the frozen run configuration and is intended to support AAAI reproducibility-checklist item 4.13. The machine-readable companion is `final-experiment-configuration.json`.

## Shared generation settings

- Temperature: `0`.
- Per-cell requested seed: `int(cell_id[:8], 16)`.
- `top_p`, `top_k`, and other sampling controls: not overridden; provider defaults.
- One generation per model-question-condition cell.
- 100 questions per dataset; question-selection seed `20260722`.
- Eight conditions: zero; system; before; after; system+before; system+after; before+after; system+before+after.
- Provider fallback after preflight: disabled; every generated cell stays on its pinned route.
- Route qualification load: 8 concurrent requests.

## Per-model serving settings

| Experiment model | Exact served model | Backend | Provider | Output ceiling | Route concurrency |
|---|---|---|---|---:|---:|
| `gemma-3-12b` | `google/gemma-3-12b-it:deepinfra` | huggingface | deepinfra | 1,280 | 16 |
| `llama-3.3-70b-instruct` | `meta-llama/llama-3.3-70b-instruct` | openrouter | deepinfra | 1,280 | 16 |
| `llama-4-scout` | `meta-llama/Llama-4-Scout-17B-16E-Instruct:deepinfra` | huggingface | deepinfra | 1,280 | 16 |
| `ministral-3-14b-2512` | `mistralai/ministral-14b-2512` | openrouter | mistral | 4,352 | 16 |
| `mistral-large-3-2512` | `mistralai/mistral-large-2512` | openrouter | mistral | 2,304 | 16 |
| `qwen3-30b-a3b-instruct-2507` | `qwen/qwen3-30b-a3b-instruct-2507` | openrouter | nebius | 2,304 | 16 |
| `qwen3-235b-a22b-instruct-2507` | `qwen/qwen3-235b-a22b-2507` | openrouter | alibaba | 2,816 | 16 |

The output ceiling is the only generation limit that differs across models. Provider-advertised maximum capability is not used as the request ceiling.

## Statistical settings

- Question-cluster bootstrap resamples: 10,000.
- Paired question-level sign-flip draws: 50,000.
- Confidence level: 95%.
- Multiplicity: Holm correction for the pooled exploratory endpoint family and across seven models within each model-specific endpoint family.
- Human audit: frozen stable-hash sampling and side assignment; exact-binomial design/test as stored in the validation artifacts.

## Lexical measurement settings

- Reference scope: `global_pubmed_abstracts_2010_2024`.
- Reference documents: 15,103,887.
- IDF formula: `log((N + 1) / (df + 1)) + 1`.
- IDF cap: `10.0`.
- High-IDF threshold: `6.0`.
- Sublinear source term frequency: enabled.
- Candidate credit is capped by source-stem frequency; exact matching/normalization rules are stored in the lexical-reference artifact and implemented in `lexical.py`.

## Local orchestration environment

- Python: `3.13.14` (`CPython`).
- Platform: `win32`; architecture: `ARM64`.
- Exact Python package versions are embedded in the machine-readable configuration and frozen run environment file.
- Hosted-provider GPU/CPU/RAM were not exposed by the providers; this is an infrastructure-visibility limitation, not an omitted local setting.

## Human-validation settings

- Audit version: `aaai27-effect-validation-v1`.
- Primary estimand: probability that a blinded human selects the mechanically preferred A/B side among automatic all-eight 0→1 improvements.
- `p0 = 0.80`, `p1 = 0.95`, one-sided `alpha = 0.05`, target power `0.80`.
- Prospective exact-binomial design: primary `n = 30`, critical confirmations `28`, achieved power `0.81217881314696` at `p1`; `cannot tell` is a non-confirmation and remains in the fixed denominator.
- Sentinels: 3 degradation + 3 tie tasks (10% of primary n for each type), descriptive/unpowered.
- Total blinded tasks: 36; 33 unique questions; 36 unique question-model units/matched pairs.
- Sampling: stable-hash deterministic random sampling, proportional across atomic role-change patterns for the primary stratum.
- Seed: `instruction-duplication-aaai27-effect-validation-2026-08-19-v1`.
- A/B orientation: deterministically randomized and treatment identity hidden. Model, dataset machine stratum, mechanical preference, and gold answer are hidden from the rater as documented in the design artifact.
- Selection does not use gold answer, response quality, score magnitude, or final human rating.
- Exact design and all frozen values are authoritative in `human-validation/aaai27-human-validation-design.json` and `human-validation/pre-rating-commitment.json`.

## Version/provenance distinction

- Frozen generation workspace manifest package version: `3.0.6`.
- The supplied source snapshot/final deterministic judging code is later (`3.0.13` provenance in the validation package) because post-generation judge refinement was performed on the already-frozen generations.
- No answers were regenerated for that refinement. This package preserves the frozen generations and the final measurement/validation artifacts separately.

## Provider-route candidate sets and selection

The complete candidate lists and backend order are authoritative in `paper-run/config/models.json` and `paper-run/config/preflight.json`. Route selection used only functional probes and the 8-concurrent-request capacity qualification, not paper outcomes. The exact selected routes are frozen in `paper-run/config/routes.json`; no provider fallback occurred after preflight.

## Protocol identity

- Exactly one procedural wording was used in the reported run.
- Protocol SHA-256: `86b92dcf90caef0528ec7270c28bcd50f65e2a9bbc187ea8348824f06001b4c1`.
- Exact text is stored in `src/instruction_duplication/protocol.py` and reproduced in `docs/DEVELOPMENT_CHOICES.md`.
