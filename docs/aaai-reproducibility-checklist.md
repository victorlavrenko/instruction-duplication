# AAAI reproducibility checklist mapping

This file maps the repository and frozen paper-run artifacts to the AAAI reproducibility checklist. It intentionally leaves the human-validation discussion separate.

## General paper structure

- 1.1 Contributions distinguished from prior work: **yes** — paper introduction/contributions and related work.
- 1.2 Limitations stated: **yes** — paper limitations/reproducibility section.
- 1.3 Assumptions and scope stated: **yes** — model panel, medical task family, visible-text scope, hosted-provider limitations.

## Theoretical contributions

- 2.1 Theoretical result: **no**. The paper is empirical; theorem/proof sub-items are not applicable.

## Datasets

- 3.1 Dataset statistics reported: **yes** — 300 questions total, 100 per benchmark; run artifact contains selected records.
- 3.2 Dataset split/source described: **yes** — exact repositories, configs, splits and immutable revisions are in `datasets_loader.py`, `docs/paper-run-parameters.md`, and the run manifest.
- 3.3 New dataset documentation: **NA** — no new dataset is introduced.
- 3.4 New dataset release: **NA**.
- 3.5 Existing datasets referenced: **yes** — paper citations plus exact machine-readable provenance.
- 3.6 Existing datasets publicly available: **yes** for the pinned sources used by this study; exact revisions are recorded.
- 3.7 Non-public datasets: **NA**.

## Computational experiments

- 4.1 Computational experiments included: **yes**.
- 4.2 Development hyperparameters and selection criteria: **partial**. There was no broad outcome-driven hyperparameter sweep. `docs/reproducibility.md` records which parameters were fixed, which were experimental factors, which settings were smoke-calibrated, and the criterion for final output ceilings/routes. Marking this `partial` is more accurate than inventing ranges that were never tried.
- 4.3 Preprocessing code included: **yes**, when the supplementary ZIP built by `scripts/build_aaai_supplement.py` is uploaded.
- 4.4 Experimental/analysis source included: **yes**, when that ZIP is uploaded.
- 4.5 Public research-usable license: **yes** — MIT.
- 4.6 New-method implementation comments with paper references: **partial/yes depending on checklist interpretation**. The implementation is extensively documented and `docs/reproducibility.md` provides an explicit source-to-paper section map; several modules also carry focused docstrings/comments. The repository does not artificially annotate every helper function with paper section numbers.
- 4.7 Random seeds described: **yes** — question selection seed and deterministic per-cell generation seed are documented; resampling procedures use fixed seeds in the analysis implementation/artifacts.
- 4.8 Computing infrastructure: **partial by unavoidable provider opacity**. Exact local software environment/platform and exact hosted model/provider routes are recorded. Hosted providers did not expose physical GPU/CPU/RAM details, so those cannot be truthfully supplied.
- 4.9 Evaluation metrics formally described and motivated: **yes** — paper plus `docs/experiment.md`.
- 4.10 Number of algorithm runs stated: **yes** — one generation per scheduled model-question-condition cell; failures/truncations retained under ITT conventions.
- 4.11 Distributional/confidence analysis: **yes** — question-clustered confidence intervals and model/dataset analyses.
- 4.12 Statistical significance tests: **yes** — paired question-level sign-flip tests with declared Holm corrections.
- 4.13 Final model/algorithm hyperparameters: **yes** — `docs/paper-run-parameters.md` plus frozen `manifest.json`, `config/models.json`, `config/routes.json`, `config/environment.json` and `config/preflight.json` in the paper-run artifact.

## Supplement contents required for the answers above

The review supplement should contain, at minimum:

1. the anonymized source snapshot;
2. `requirements-research.lock` and `pyproject.toml`;
3. `docs/experiment.md`, `docs/reproducibility.md`, `docs/paper-run-parameters.md`, and `docs/THIRD_PARTY.md`;
4. the frozen paper-run `manifest.json`;
5. `paper-run/config/` (especially models, routes, preflight, environment);
6. `paper-run/data/` (selected questions, schedule, lexical reference/QC artifacts);
7. `paper-run/results/` needed to reproduce the reported analysis;
8. `paper-run/state/run.sqlite3` if the submission size limit permits the complete rejudge/reanalyze workflow;
9. SHA-256 checksums and a short anonymous README.

The supplied builder includes all of the above by default and intentionally withholds the human-audit treatment/mechanical decoding key until human validation is finalized.
