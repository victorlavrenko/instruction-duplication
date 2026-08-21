# Instruction Duplication

Instruction Duplication tests a simple inference-time control mechanism: repeat the same procedural instruction and measure whether the model follows it more completely.

The experiment evaluates the same medical multiple-choice questions under eight instruction-placement conditions. The default full run uses:

- 300 questions: 100 each from MedQA, MedXpertQA, and AfriMed-QA
- 7 instruction-tuned language models
- 8 prompt conditions
- 16,800 scheduled generations

The main question is whether duplication changes the **trajectory the model produces**, separately from whether its final answer is correct.

## Install

Python 3.12.13 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.lock
python -m pip install -e .
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproduce the experiment

Generation uses Hugging Face and/or OpenRouter. Set the credentials for the providers you want to use:

```bash
export HF_TOKEN=hf_...
export OPENROUTER_API_KEY=...
```

PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
$env:OPENROUTER_API_KEY = "..."
```

Run the default 100-question-per-dataset experiment:

```bash
instruction-duplication 100 --workspace run
```

For a small smoke run:

```bash
instruction-duplication 10 --workspace run-smoke
```

The pipeline prepares the selected questions, resolves and pins provider routes, generates missing cells, judges the stored responses, and writes the statistical analysis. It is resumable: completed cells are reused when the experiment identity still matches.

Use an explicit spending limit when desired:

```bash
instruction-duplication 10 \
  --workspace run-smoke \
  --max-cost 20
```

The individual stages are also available:

```bash
instruction-duplication prepare 10 --workspace run
instruction-duplication preflight --workspace run
instruction-duplication run --workspace run --max-cost 20
instruction-duplication status --workspace run
instruction-duplication judge --workspace run
instruction-duplication analyze --workspace run
```

See [docs/providers.md](docs/providers.md) for provider selection, retry behavior, concurrency, and budgeting.

## Reanalyze the paper run

The complete frozen generation used for the paper is distributed separately from the Git repository because the generated responses and run database are large research artifacts rather than source code.

After extracting the archive, the stored generations can be rejudged and reanalyzed without provider credentials:

```bash
instruction-duplication status --workspace run-2026-08-12
instruction-duplication judge --workspace run-2026-08-12
instruction-duplication analyze --workspace run-2026-08-12
```

`judge` operates only on the stored questions and generations. It does not regenerate responses or contact an inference provider.

`analyze` recomputes the paired statistical analysis from the current deterministic judgments.

The frozen run archive will be published as a separate release artifact together with the paper.

For the exact August 12 paper-run model IDs, provider routes, output ceilings, seeds, dataset revisions, and statistical settings, see [docs/paper-run-parameters.md](docs/paper-run-parameters.md). For the complete final parameter appendix, development-choice record, and paper-to-code map, see [docs/FINAL_EXPERIMENT_CONFIGURATION.md](docs/FINAL_EXPERIMENT_CONFIGURATION.md), [docs/DEVELOPMENT_CHOICES.md](docs/DEVELOPMENT_CHOICES.md), and [docs/PAPER_IMPLEMENTATION_MAP.md](docs/PAPER_IMPLEMENTATION_MAP.md). The shortest reproduction path is [REPRODUCE.md](REPRODUCE.md); the broader AAAI reproducibility rationale is in [docs/reproducibility.md](docs/reproducibility.md).

For double-blind review, do **not** link this author-identifying public repository from the manuscript. Build an anonymized supplementary code/data ZIP from a local checkout plus the frozen run:

```bash
python scripts/build_aaai_supplement.py \
  --run /path/to/run-2026-08-12 \
  --output aaai27-supplement.zip
```

The builder copies the source and paper-run workspace, scrubs known author-identifying strings from text metadata, withholds the human-audit decoding key until ratings are finalized, writes SHA-256 checksums, and fails if known identifiers remain. The resulting ZIP must still be inspected manually before upload.

## Experimental design

The instructed conditions ask the model to perform eight reasoning roles in order:

1. Facts
2. Implications
3. Provisional answer
4. Best alternative
5. Decisive distinction
6. What would change the answer
7. Reconsideration
8. Final answer

The eight prompt conditions vary instruction copy count and placement:

- no instruction
- system only
- before the question
- after the question
- system + before
- system + after
- before + after
- system + before + after

The principal copy-count comparison contrasts the three one-copy conditions with the three two-copy conditions. Placement effects are analyzed separately.

See [docs/experiment.md](docs/experiment.md) for the exact protocol, contrasts, and analysis conventions.

## Measurements

Final-answer accuracy and trajectory quality are measured separately.

The main trajectory measurements are:

- **Substantive protocol completion** — whether the requested reasoning roles are present and actually perform their requested functions.
- **Pre-provisional TF-IDF recall** — how much question-specific material is exposed before the model commits to a provisional answer.
- **Contrastive discussion** — whether the response supplies a provisional answer, a different alternative, a decisive distinction, an answer-changing counterfactual, and reconsideration.
- **Accuracy** — correctness of the final multiple-choice answer.

TF-IDF uses global document frequencies from a pinned PubMed reference corpus of 15,103,887 abstracts from 2010–2024. Credit is capped by the source stem, so repeating the same term cannot increase recall indefinitely.

These lexical measurements quantify **visible question-specific coverage**. They are not measures of hidden reasoning, proposition truth, or medical correctness.

The judge also records structural and lexical diagnostics such as section order, section substance, polarity/laterality/timing anchors, response length, and TF-IDF mass per 100 pre-answer tokens.

## Human validation

Automatic measurements are designed to be auditable. The run exports condition-blinded matched pairs for simple human checks of individual judge components.

The validation tasks are intentionally atomic. Reviewers are not asked to solve the medical question or decide whether a model's diagnosis is correct. For example, a lexical-coverage task highlights question-specific terms present in each response and asks which response preserves more of them.

Human disagreement is analyzed as a signed correction to the automatic treatment effect rather than requiring perfect human-machine agreement.

The final human-validation protocol and results are intentionally treated separately from the automatic reproduction package until ratings are frozen.

## Outputs

A workspace has the following structure:

```text
run/
├── manifest.json
├── config/
├── data/
├── state/
│   └── run.sqlite3
└── results/
    ├── paper-report.txt
    ├── report.txt
    ├── analysis.json
    ├── model-effect-summary.csv
    ├── model-effects.csv
    ├── cells-and-judgments.jsonl
    ├── attempts.jsonl
    ├── blinded-matched-pair-audit.jsonl
    └── blinded-matched-pair-key.jsonl
```

The most useful outputs are:

- `paper-report.txt` — compact results intended for the paper
- `report.txt` — full analysis and audit report
- `analysis.json` — machine-readable statistical results
- `model-effects.csv` — model-by-metric effects, confidence intervals, and p-values
- `cells-and-judgments.jsonl` — cell-level responses and deterministic judgments

`manifest.json` binds the selected questions, model panel, protocol, scorer versions, and prepared environment. Provider routes selected during preflight are stored under `config/`.

The SQLite database and large JSONL exports are runtime/research artifacts and are not intended to be committed to ordinary Git history.

## Independent question selection

A new experiment can exclude questions already used in another workspace:

```bash
instruction-duplication 100 \
  --workspace run-confirmatory \
  --exclude-workspace run-pilot \
  --exclude-workspace run-smoke
```

`--exclude-workspace` is repeatable. Exclusion uses both source-qualified question IDs and normalized stems to avoid silently reusing the same question.

## Development

Install development and test dependencies:

```bash
python -m pip install -e ".[dev,test]"
```

Then run the test suite and see [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/development.md](docs/development.md).

## AAAI-27 frozen robustness reanalysis

The paper-facing robustness checks added after the judge was frozen do **not** regenerate or rejudge any response. They operate on `results/cells-and-judgments.jsonl`, the frozen question QC file, and the already frozen human-audit artifacts:

```bash
PYTHONPATH=src python tools/aaai27_robustness_analysis.py \
  --workspace /path/to/run \
  --human-dir /path/to/human-validation \
  --out /path/to/aaai27-robustness-analysis.json
```

The tool reports the full eight-condition factorial, additive higher-copy residuals, pooled and trailing-copy TF-IDF length adjustments, split generation-status diagnostics, raw accuracy numerators/denominators, TF-IDF eligibility provenance, and human-audit sign/orientation checks. These are post-hoc robustness analyses on the frozen 3.0.13 judgments; they are not changes to the generation protocol or deterministic judge.
