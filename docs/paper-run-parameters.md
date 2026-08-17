# Frozen August 12, 2026 paper-run parameters

This file records the final generation parameters and exact provider routes used by the paper run. It is descriptive provenance for `run-2026-08-12`; it is not regenerated from current provider availability.

## Shared generation settings

- temperature: `0`
- seed: deterministic per generation cell, `int(cell_id[:8], 16)`
- provider fallback after preflight: disabled; every generated cell stays on its pinned route
- route qualification load: 8 concurrent requests
- question-selection seed: `20260722`
- questions per dataset: 100
- datasets: MedQA, MedXpertQA, AfriMed-QA
- prompt conditions: zero, system, before, after, system+before, system+after, before+after, system+before+after
- main duplication contrast: mean of the three two-copy conditions minus mean of the three one-copy conditions, paired within model-question blocks
- generation package version: `3.0.6`
- source/judge analysis line after rejudging: `3.0.9`

## Exact model and route table

| Experiment model | Exact served model | Backend | Provider | Output ceiling | Max route concurrency |
|---|---|---|---|---:|---:|
| Gemma 3 12B | `google/gemma-3-12b-it:deepinfra` | Hugging Face | DeepInfra | 1,280 | 16 |
| Llama 3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | DeepInfra | 1,280 | 16 |
| Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct:deepinfra` | Hugging Face | DeepInfra | 1,280 | 16 |
| Ministral 3 14B Instruct | `mistralai/ministral-14b-2512` | OpenRouter | Mistral | 4,352 | 16 |
| Mistral Large 3 | `mistralai/mistral-large-2512` | OpenRouter | Mistral | 2,304 | 16 |
| Qwen3 30B-A3B Instruct | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter | Nebius | 2,304 | 16 |
| Qwen3 235B-A22B Instruct | `qwen/qwen3-235b-a22b-2507` | OpenRouter | Alibaba | 2,816 | 16 |

The corresponding versioned model identities were:

| Experiment model | Hugging Face ID | OpenRouter ID |
|---|---|---|
| Gemma 3 12B | `google/gemma-3-12b-it` | `google/gemma-3-12b-it` |
| Llama 3.3 70B Instruct | `meta-llama/Llama-3.3-70B-Instruct` | `meta-llama/llama-3.3-70b-instruct` |
| Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `meta-llama/llama-4-scout` |
| Ministral 3 14B Instruct | `mistralai/Ministral-3-14B-Instruct-2512` | `mistralai/ministral-14b-2512` |
| Mistral Large 3 | `mistralai/Mistral-Large-3-675B-Instruct-2512` | `mistralai/mistral-large-2512` |
| Qwen3 30B-A3B Instruct | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `qwen/qwen3-30b-a3b-instruct-2507` |
| Qwen3 235B-A22B Instruct | `Qwen/Qwen3-235B-A22B-Instruct-2507` | `qwen/qwen3-235b-a22b-2507` |

## Dataset revisions

| Dataset | Repository/config | Split used | Immutable revision |
|---|---|---|---|
| MedQA | `GBaker/MedQA-USMLE-4-options` | test | `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18` |
| MedXpertQA | `TsinghuaC3I/MedXpertQA` / `Text` | test | `7e7c465a68eb2b866926bfa59c8c9d17a8daba65` |
| AfriMed-QA | `afrimedqa/afrimedqa_v2` | train, internal test partition | `3b4382fa0bb51bfc026f5813021ab0ec7be9de8f` |

## Statistical settings

- confidence intervals: 10,000 question-cluster bootstrap resamples
- p-values: 50,000 paired question-level sign-flip draws
- model-specific multiplicity: Holm correction across seven models within endpoint
- key secondary multiplicity: Holm correction within the declared endpoint family
- generation failures: retained under the paper's intention-to-treat conventions

## Environment snapshot

The frozen run records CPython `3.13.14`, Windows (`win32`) on `ARM64`. Exact package versions are stored in `run-2026-08-12/config/environment.json`. Hosted-provider physical GPU/CPU/RAM details were not exposed by the providers and therefore cannot be reconstructed reliably.

## Authoritative machine-readable provenance

When available, the files inside the frozen workspace supersede this human-readable summary:

- `manifest.json`
- `config/models.json`
- `config/routes.json`
- `config/environment.json`
- `config/preflight.json`
- `data/generation-schedule.json`

These are included in the intended supplementary artifact so that reviewers do not need to rely on this table alone.
