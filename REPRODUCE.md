# Reproduce the reported results

These commands are for a normal GitHub checkout of this repository. The frozen paper run and human-validation artifacts are distributed separately in the AAAI replication/validation package; **do not regenerate answers** to reproduce the reported analysis.

## 1. Install the repository

From the repository root:

```bash
python -m pip install -e ".[test]"
```

No provider credentials are needed for rejudging, analysis, robustness analysis, or human-validation scoring.

## 2. Point to the extracted validation package

Extract the frozen replication/validation package somewhere outside the Git checkout. In the commands below, replace `/path/to/validation-package` with that directory.

For POSIX shells:

```bash
VALIDATION=/path/to/validation-package
```

For PowerShell:

```powershell
$VALIDATION = "C:\path\to\validation-package"
```

The directory must contain `paper-run/`, `human-validation/`, and `robustness-analysis/`.

## 3. Rejudge the existing frozen generations

POSIX:

```bash
instruction-duplication judge --workspace "$VALIDATION/paper-run"
```

PowerShell:

```powershell
instruction-duplication judge --workspace "$VALIDATION\paper-run"
```

Expected primary output: `paper-run/results/cells-and-judgments.jsonl` plus the workspace's normal deterministic judge/report artifacts. This command reads stored generations only; it does not contact an inference provider.

## 4. Reproduce the paper analysis

POSIX:

```bash
instruction-duplication analyze --workspace "$VALIDATION/paper-run"
```

PowerShell:

```powershell
instruction-duplication analyze --workspace "$VALIDATION\paper-run"
```

Expected outputs are the analysis/report artifacts under `paper-run/results/`, including `analysis.json`, `paper-report.txt`, `report.txt`, `model-effect-summary.csv`, and `model-effects.csv`.

## 5. Run the AAAI robustness analysis

POSIX:

```bash
python tools/aaai27_robustness_analysis.py \
  --workspace "$VALIDATION/paper-run" \
  --human-dir "$VALIDATION/human-validation" \
  --out "$VALIDATION/robustness-analysis/reproduced-aaai27-robustness-analysis.json"
```

PowerShell:

```powershell
python tools/aaai27_robustness_analysis.py `
  --workspace "$VALIDATION\paper-run" `
  --human-dir "$VALIDATION\human-validation" `
  --out "$VALIDATION\robustness-analysis\reproduced-aaai27-robustness-analysis.json"
```

Expected primary output: `robustness-analysis/reproduced-aaai27-robustness-analysis.json`. The existing CSV/JSON files in `robustness-analysis/` are the frozen reported outputs.

## 6. Score the frozen human validation

POSIX:

```bash
python tools/aaai27_human_validation.py score \
  --ratings "$VALIDATION/human-validation/aaai27-human-validation-ratings-42f6b76cd152eade595d329c.json" \
  --key "$VALIDATION/human-validation/aaai27-human-validation-key.jsonl" \
  --design "$VALIDATION/human-validation/aaai27-human-validation-design.json" \
  --out-dir "$VALIDATION/human-validation/reproduced-score"
```

PowerShell:

```powershell
python tools/aaai27_human_validation.py score `
  --ratings "$VALIDATION\human-validation\aaai27-human-validation-ratings-42f6b76cd152eade595d329c.json" `
  --key "$VALIDATION\human-validation\aaai27-human-validation-key.jsonl" `
  --design "$VALIDATION\human-validation\aaai27-human-validation-design.json" `
  --out-dir "$VALIDATION\human-validation\reproduced-score"
```

Expected outputs in `human-validation/reproduced-score/` are the scored JSON/text/LaTeX reports. Frozen reported counterparts are `aaai27-human-validation-report.json`, `aaai27-human-validation-report.txt`, and `aaai27-human-validation-result.tex`.

## 7. Run the repository test suite

```bash
pytest
```

If the package has not been installed editable first, the equivalent source-tree command is:

```bash
PYTHONPATH=src python -m pytest
```

Configuration/provenance is documented in `docs/FINAL_EXPERIMENT_CONFIGURATION.md`, `docs/final-experiment-configuration.json`, `docs/DEVELOPMENT_CHOICES.md`, `docs/PAPER_IMPLEMENTATION_MAP.md`, and `docs/PROVIDER_INFRASTRUCTURE_VISIBILITY.md`.
