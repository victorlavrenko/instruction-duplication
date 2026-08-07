# Development guide

## Repository layout

The package uses a `src` layout. Production code lives in `src/instruction_duplication`, tests in `tests`, repository-specific checks in `tests/coding_quality`, and release verification in `tools`.

The main modules have deliberately narrow responsibilities:

- `manifest.py` builds and validates the complete experiment identity.
- `datasets_loader.py` loads pinned sources and normalizes multiple-choice rows.
- `protocol.py` defines the eight conditions and the authoritative XML protocol.
- `preflight.py` discovers and probes provider routes.
- `provider.py` builds requests, validates responses, and interprets retry information.
- `generate.py` schedules fair per-model concurrent generation, provider gates, retries, and cost reservations.
- `storage.py` owns SQLite transactions, constraints, attempts, and judgments.
- `judge.py` parses responses and computes protocol outcomes.
- `lexical.py` computes deterministic stem-detail coverage measures.
- `stats.py` performs paired analysis, confidence intervals, endpoint-scoped model multiplicity correction, and heterogeneity checks.
- `report.py` renders the text report.
- `workspace.py` handles paths, hashes, atomic writes, and process locking.
- `cli.py` composes these modules into stage commands and the `reproduce` workflow.

## Data boundaries

Provider responses, decoded JSON, and dynamically loaded dataset rows enter the program as `object`. They must be validated before they are used as mappings, sequences, or domain records. Keep recursive JSON types confined to I/O boundaries; application logic should use concrete dataclasses and enums.

The codebase uses strict mypy and Pyright configurations. Do not solve typing errors with `Any`, `cast`, `type: ignore`, `pyright: ignore`, or `noqa`. Add a validator, protocol, typed factory, or explicit domain type instead.

## Runtime invariants

A prepared workspace can be resumed for generation only when its manifest identity and prepared runtime environment match the current request. Read-only status, judging, and analysis validate the stored environment hash but do not require the reader to reproduce the original machine environment. The manifest is written last during preparation. Route selection is bound to that manifest, and judgments are bound to the response hash and scorer versions.

Each provider request has a logical idempotency key and a separate attempt record. Transport retries with the same payload reuse the logical key; a truncation retry with a larger output ceiling uses a new key because the payload changed. Explicit HTTP 429 rejections release their reservations, while failures with uncertain provider execution remain conservatively accounted. Transient failures are requeued into later sweeps and reduce exact-route concurrency; sustained success restores it gradually. Truncated output is retried with a larger ceiling up to provider capability. Cost reservations use each attempt's actual requested ceiling rather than the provider maximum.

The XML parser and the reported compliance outcome share one validity definition. Answer extraction remains separate, so a malformed protocol response can still contribute to intention-to-treat accuracy without receiving compliance credit.

## Tests

Tests are grouped by the behavior they cover:

- answer and choice normalization: `test_answer_utils.py`, `test_datasets.py`;
- protocol rendering and XML parsing: `test_io_models_protocol.py`, `test_judge.py`;
- provider responses and route selection: `test_provider.py`, `test_preflight.py`;
- generation, retries, budgets, and storage: `test_generate.py`, `test_storage.py`;
- statistical analysis: `test_stats.py`;
- workspace identity and locking: `test_workspace.py`;
- command-line workflows: `test_cli.py`;
- repository configuration and source rules: `tests/coding_quality`.

Use the fake provider and JSONL fixture for end-to-end tests. Unit and integration tests should not require network access or provider credentials.

## Local checks

```bash
python -m pip install -e ".[dev,test]"
python tests/coding_quality/quality_gate.py
python -m ruff check .
python -m ruff format --check .
python -m pyright
python -m mypy
coverage run -m pytest
coverage report
python -m build
python tools/verify_release.py
```

The CI matrix runs tests on the minimum supported Python and newer supported interpreters on Linux and Windows. Static analysis uses the exact versions in `requirements-dev.lock`. The package job builds both distributions, installs the wheel, and runs a network-free experiment from the installed package.

## Supported baseline

The project requires Python 3.12.13 or newer and the dependency ranges declared in `pyproject.toml`. There are no compatibility branches for older Python releases, older dependency APIs, or older workspace schemas. Start a new workspace when the stored schema is not accepted by the current package.
