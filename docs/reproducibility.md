# Reproducibility notes for the AAAI-27 paper

This document centralizes details requested by the AAAI reproducibility checklist. It describes the August 12, 2026 paper run and distinguishes experimental factors from engineering settings.

## Development and parameter selection

The study is not the result of a broad accuracy-driven hyperparameter search. The experimental factors are instruction copy count and placement; they are deliberately varied rather than tuned.

Generation settings were frozen before the paper run:

- temperature: `0` for every generation; no temperature sweep;
- seed: deterministic per cell, `int(cell_id[:8], 16)`; no seed selection by outcome;
- sampling parameters other than temperature/seed: provider defaults; no top-p/top-k sweep;
- questions: 100 per dataset, selected with seed `20260722` by the frozen normalization/deduplication/shuffle procedure;
- model panel: seven fixed instruction-tuned models selected before the paper run to cover multiple model families and sizes;
- prompt: one frozen eight-role protocol; copy count and placement are the experimental factors, not optimized prompt variants;
- output ceilings: model-specific loop-bounding ceilings chosen from earlier smoke-run output-length observations, with a targeted Mistral Large correction after a current-protocol smoke showed the old ceiling was too close to an ordinary completion tail;
- route selection: provider routes were selected by functional and concurrent-capacity preflight probes, not by task accuracy or compliance;
- bootstrap resamples: 10,000;
- paired sign-flip draws: 50,000;
- multiple-testing correction: Holm within the declared endpoint family and, for model-specific analyses, across the seven models within each endpoint.

The measurement implementation was refined after generation. Consequently, the August 12 run is reported as exploratory/judge-development evidence rather than a preregistered confirmatory test. No generation was replaced or regenerated because of the later judge changes.

## Exact paper-run identity

The frozen generation workspace records:

- generation package version: `3.0.6`;
- current source/judge analysis line: `3.0.9`;
- question-selection seed: `20260722`;
- question set: 300 total, 100 each from MedQA, MedXpertQA and AfriMed-QA;
- conditions: eight placements over system/before-query/after-query locations;
- scheduled generations: 16,800;
- manifest question hash: `07bf8a509b0d6d26d652123037b2b980e0d1f6157e80003e3ab69fd724a9698a`;
- manifest protocol hash: `86b92dcf90caef0528ec7270c28bcd50f65e2a9bbc187ea8348824f06001b4c1`;
- manifest condition hash: `62b4f578242cfd97ecc81b5ced27db652489adec9430f4398cd9dc6c2ddf43c1`.

See [`paper-run-parameters.md`](paper-run-parameters.md) for the exact model IDs, routes and output ceilings used in that run.

## Computing infrastructure

The local orchestration environment recorded by the paper workspace was:

- Python `3.13.14` (CPython);
- platform `win32`;
- machine architecture `ARM64`;
- `instruction-duplication==3.0.6` at generation time;
- exact dependency versions are stored in `run-2026-08-12/config/environment.json` in the frozen run artifact.

Generation itself used hosted Hugging Face Inference Providers and OpenRouter routes. Those services did not expose the physical serving GPU model, GPU memory, host CPU, or host RAM. The experiment therefore cannot truthfully report unavailable hardware attributes. Instead it records the exact backend, upstream provider, model route, request contract, concurrency qualification, token ceilings, attempts, latency and provider-reported usage/cost. This limitation is stated explicitly rather than inferring hardware.

## Dataset provenance

Dataset repositories, splits and immutable revisions are pinned in `src/instruction_duplication/datasets_loader.py` and copied into every experiment manifest. The August 12 run used:

- MedQA: `GBaker/MedQA-USMLE-4-options`, test split, revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`;
- MedXpertQA: `TsinghuaC3I/MedXpertQA`, `Text` config, test split, revision `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`;
- AfriMed-QA: `afrimedqa/afrimedqa_v2`, train split / internal test partition, revision `3b4382fa0bb51bfc026f5813021ab0ec7be9de8f`.

The PubMed document-frequency resource used by the lexical metric is independently pinned in [`THIRD_PARTY.md`](THIRD_PARTY.md), including upstream commit and SHA-256.

## Source-to-paper implementation map

The main implementation entry points are intentionally mapped to paper methods:

- `protocol.py` — eight-role protocol and eight placement conditions (paper: Experiment / Protocol and Conditions);
- `datasets_loader.py` — frozen dataset sources and question selection (paper: Models, Questions, and Conditions);
- `provider.py` and `models.py` — deterministic request settings, exact model IDs and output ceilings (paper: Generation and Reproducibility);
- `trajectory.py` — format-tolerant recovery of the requested visible roles (paper: Measurement);
- `judge.py` — structural/substantive protocol and contrastive diagnostics (paper: Measurement);
- `lexical.py` and `pubmed_idf.py` — pre-provisional TF-IDF exposure metric (paper: Measurement / TF-IDF);
- `stats.py` — paired, question-clustered bootstrap/sign-flip inference and multiplicity handling (paper: Inference);
- `audit.py` — blinded matched-pair exports used by the human-validation workflow (paper: Evaluation Validity).

## Reproducing without new generation

The frozen paper run can be rejudged and reanalyzed without provider credentials:

```bash
instruction-duplication status --workspace run-2026-08-12
instruction-duplication judge --workspace run-2026-08-12
instruction-duplication analyze --workspace run-2026-08-12
```

The `judge` command reads stored questions and generations only. `analyze` recomputes the paired statistical outputs from stored deterministic judgments.

## Anonymous supplementary package

For double-blind review, do not link the public author-identifying Git repository from the manuscript. Build an anonymous snapshot with:

```bash
python scripts/build_aaai_supplement.py \
  --run /path/to/run-2026-08-12 \
  --output aaai27-supplement.zip
```

The builder copies the reproduction source and frozen run artifacts, removes files that are unnecessary for reproduction, scrubs known author-identifying strings from text metadata, and fails if those strings remain. Inspect the resulting archive manually before uploading it to OpenReview.

Human-validation results are intentionally not documented here until that protocol is finalized and ratings are frozen.
