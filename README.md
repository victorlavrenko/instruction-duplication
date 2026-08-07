# Instruction Duplication

Instruction Duplication runs a controlled experiment on instruction placement and repetition. It compares the same multiple-choice questions across eight prompt conditions, from no protocol to three copies placed in the system message and before or after the question.

The default configuration uses seven instruction-tuned models and three medical question datasets. A full run with 100 questions per dataset contains 300 questions × 7 models × 8 conditions, or 16,800 cells.

## Analyze the committed run

The repository is intended to include a completed `run/` workspace. You can inspect and reanalyze that generation without provider credentials and without sending any model requests.

Create an environment from the checked-out source:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.lock
python -m pip install -e .
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Inspect the committed run:

```bash
instruction-duplication status --workspace run
instruction-duplication analyze --workspace run
```

`analyze` recomputes the statistical analysis from the stored cells and judgments. It does not contact inference providers. Pooled endpoints are reported individually without joint Holm correction; model-specific tests are Holm-adjusted across the seven models separately within each endpoint. Trailing-copy and dataset-specific contrasts are reported unadjusted as exploratory or descriptive analyses. The main outputs are:

- `run/results/report.txt` — readable analysis summary;
- `run/results/analysis.json` — machine-readable statistical results;
- `run/results/cells-and-judgments.jsonl` — flat cell-level export;
- `run/results/attempts.jsonl` — complete preflight and generation attempt provenance.

If scorer code changes while the stored generations remain the intended sample, refresh deterministic judgments first:

```bash
instruction-duplication judge --workspace run
instruction-duplication analyze --workspace run
```

Judging and analysis use the stored generation data and require no provider API keys.

A full run can exceed GitHub's ordinary per-file size limit, especially `run/state/run.sqlite3`. This repository marks the database and large JSONL exports for Git LFS. Clone with Git LFS enabled so those files are materialized before running `status`, `judge`, or `analyze`.

## Reproduce the generation

A new generation requires Python 3.12.13 or newer, a Hugging Face token, an OpenRouter API key, or both, and enough provider credit for the requested run. With the default `--backend auto` policy, each model has its own preferred backend and preflight falls back to the other backend only if the preferred one is unavailable or fails its probes. The default panel keeps Hugging Face preferred for four models and prefers OpenRouter for `llama-3.3-70b-instruct`, `qwen3-30b-a3b-instruct-2507`, and `qwen3-235b-a22b-instruct-2507`. These three models showed the clearest provider-side throughput constraints in observed runs; the Qwen3 235B default avoids the Hugging Face/DeepInfra route that repeatedly timed out and collapsed to single-request concurrency.

Set credentials for the providers you want preflight to consider:

```bash
export HF_TOKEN=hf_...
export OPENROUTER_API_KEY=...
```

PowerShell equivalents:

```powershell
$env:HF_TOKEN = "hf_..."
$env:OPENROUTER_API_KEY = "..."
```

Run the full 100-question-per-dataset replication into a fresh workspace:

```bash
instruction-duplication reproduce --workspace run
```

For a smaller smoke or exploratory run:

```bash
instruction-duplication reproduce 10 --workspace run
```

Per-model defaults are part of the versioned model panel. `--backend hf` or `--backend openrouter` still forces one backend for every model. Under the default `--backend auto`, a single model can be overridden without changing the source panel:

```bash
instruction-duplication reproduce 10 --workspace run \
  --prefer-backend llama-3.3-70b-instruct=hf \
  --prefer-backend qwen3-30b-a3b-instruct-2507=openrouter
```

`--prefer-backend` changes only preflight order. Once `config/routes.json` exists, the selected exact routes remain pinned; use a fresh workspace or run `preflight` explicitly before generation if you want different routes.

`reproduce` prepares the workspace, pins one provider route per model, generates missing cells, refreshes deterministic judgments, and writes the analysis. It can be rerun after interruption. Existing cells are reused only when the experiment identity and pinned route provenance still match.

Transient provider failures are retried in later sweeps rather than being exhausted immediately on one cell. A transient error reduces concurrency for the affected exact route and provider cooldowns are shared by workers using that provider; successful requests restore route concurrency gradually. The selected provider route remains pinned throughout generation.

If a provider ends a response because the requested output ceiling was reached, the runner records that attempt and retries the same cell with a larger ceiling, up to the model/provider capability. The ordinary ceilings remain based on the observed smoke run, so normal requests do not reserve the provider maximum unnecessarily.

After the configured transient retry rounds are exhausted, `reproduce` still judges and analyzes the run if some cells remain `retryable`. Those cells are included in the intention-to-treat analysis as generation failures, exactly like other unsuccessful generations. A later `reproduce` invocation can retry them and then refresh the judgments and analysis. The workflow stops only when cells were not fully attempted at all (`pending`, `running`, or `budget_blocked`); resolve those states or increase the cost cap before analysis.

Operational progress is written to standard error: pipeline stages, route probes and selections, generation parallelism, retry sweeps, and periodic completion counts. Generation progress includes a recent-throughput rate and ETA; the ETA is refreshed every 30 seconds during long cooldowns so adaptive route throttling is reflected instead of leaving the initial estimate unchanged. When throttling is active, a separate `bottlenecks` line names each constrained model and pinned backend/provider, its current adaptive limit versus the unconstrained worker demand, the number of active requests, and any remaining provider cooldown. The final report remains on standard output.

Use an explicit cost limit when desired:

```bash
instruction-duplication reproduce 10 \
  --workspace run \
  --max-cost 20 \
  --concurrency 64 \
  --per-model-concurrency 8
```

The runner reserves a conservative worst-case amount before dispatching each request and reconciles the reservation after the attempt. Explicit HTTP 429 rejections release their reservation because no generation was accepted; timeouts and other failures with uncertain provider execution remain conservatively accounted.

## Run individual stages

```bash
instruction-duplication prepare 10 --workspace run
instruction-duplication preflight --workspace run
instruction-duplication run --workspace run --max-cost 20
instruction-duplication status --workspace run
instruction-duplication judge --workspace run
instruction-duplication analyze --workspace run
```

The staged commands are useful when route selection, generation, or analysis needs to be inspected separately.

## Workspace layout

The path passed to `--workspace` is the workspace itself. Runtime state and exported results use ordinary subdirectories:

```text
run/
├── manifest.json
├── config/
│   ├── models.json
│   ├── environment.json
│   ├── routes.json
│   └── preflight.json
├── data/
│   ├── questions.jsonl
│   ├── dataset-audit.json
│   └── lexical-reference.json
├── state/
│   └── run.sqlite3
└── results/
    ├── analysis.json
    ├── report.txt
    ├── cells-and-judgments.jsonl
    └── attempts.jsonl
```

`manifest.json` binds the selected questions, model panel, protocol, scorer versions, and prepared environment. `config/routes.json` binds the exact routes selected by preflight. The SQLite database is the authoritative mutable run state; the files under `results/` are reproducible exports.

## Experiment details

The instructed conditions require a single XML response containing facts, implications, a provisional answer, a contrastive check, rereasoning, and a final answer. Compliance is measured separately from answer accuracy. Generation failures remain in the intention-to-treat analysis.

The lexical measurements are deterministic anchor-recall diagnostics. They measure surface coverage of stem details such as polarity, laterality, timing, and quantities; they are not semantic-equivalence or medical-validity scores.

See [docs/experiment.md](docs/experiment.md) for the condition design, measurements, and analysis conventions. Provider selection, retry behavior, and budgeting are described in [docs/providers.md](docs/providers.md).

## Development

Install development and test dependencies:

```bash
python -m pip install -e ".[dev,test]"
```

Then follow [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/development.md](docs/development.md).
