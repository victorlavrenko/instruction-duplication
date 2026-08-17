# Development guide

## Repository layout

The package uses a `src` layout. Production code lives in `src/instruction_duplication`, tests in `tests`, repository-specific checks in `tests/coding_quality`, and release verification in `tools`.

The main modules have deliberately narrow responsibilities:

- `manifest.py` builds and validates the complete experiment identity.
- `datasets_loader.py` loads pinned sources and normalizes multiple-choice rows.
- `protocol.py` defines the eight conditions and the authoritative labeled-section protocol.
- `preflight.py` discovers and probes provider routes.
- `provider.py` builds requests, validates responses, and interprets retry information.
- `generate.py` schedules fair per-model concurrent generation, provider gates, retries, and cost reservations.
- `storage.py` owns SQLite transactions, constraints, attempts, and judgments.
- `judge.py` parses responses and computes protocol outcomes.
- `lexical.py` computes frozen stem TF-IDF exposure and polarity-aware anchor diagnostics.
- `stats.py` performs paired analysis, confidence intervals, endpoint-scoped model multiplicity correction, and heterogeneity checks.
- `report.py` renders the text report.
- `workspace.py` handles paths, hashes, atomic writes, and process locking.
- `cli.py` composes these modules into stage commands and the `reproduce` workflow.

## Data boundaries

Provider responses, decoded JSON, and dynamically loaded dataset rows enter the program as `object`. They must be validated before they are used as mappings, sequences, or domain records. Keep recursive JSON types confined to I/O boundaries; application logic should use concrete dataclasses and enums.

The codebase uses strict mypy and Pyright configurations. Do not solve typing errors with `Any`, `cast`, `type: ignore`, `pyright: ignore`, or `noqa`. Add a validator, protocol, typed factory, or explicit domain type instead.

## Runtime invariants

Generation and deterministic measurement have deliberately different integrity requirements. `Workspace.require_prepared()` is strict about protocol, conditions, model panel, scorer versions, manifest identity, and the prepared runtime environment. Cross-version generation compatibility is explicitly enumerated. Versions 3.0.1–3.0.3 form one transport-compatible series. Version 3.0.4 changed the Mistral Large request ceiling and therefore starts a new generation identity; 3.0.4–3.0.5 form a second transport-compatible series because 3.0.5 changes only transient scheduling and new-workspace provider preference, while an existing workspace keeps its stored model configuration and pinned routes. `Workspace.require_generation_integrity()` validates immutable generation identity without requiring the original execution machine, allowing a completed run to be rejudged by a newer deterministic scorer.

The manifest is written last during preparation. Route selection is bound to that manifest, and each judgment is bound to the response hash and lexical measurement version. `analyze` refuses stale judgments, so a measurement-version change requires deterministic `judge` before statistical analysis. Rejudging may replace only derived measurement artifacts; it must not alter generation attempts, selected routes, token accounting, or provider spend.

Each provider request has a logical idempotency key and a separate attempt record. Transport/provider retries with the same payload reuse the logical key. A new preflight invocation receives a fresh qualification-run identifier, so retrying preflight preserves earlier attempts without reusing their globally unique physical request keys or provider idempotency keys. Explicit HTTP 429 rejections release their reservations, while failures with uncertain provider execution remain conservatively accounted. Isolated transient failures are requeued into later sweeps and reduce exact-route concurrency; sustained success restores it gradually. A concentrated exact-route 429 burst opens a circuit, preserves attempted and untouched states, and aborts the invocation before a serialized retry tail. Provider `finish_reason=length` is terminal: truncated output is never conditionally resampled with a larger ceiling. Cost reservations use each attempt's actual requested ceiling rather than the provider maximum.

The trajectory parser recognizes a frozen set of visible role headings rather than requiring one serialization syntax. Each scientific section extends from its recognized role heading to the next recognized role heading. Plain, numbered, Markdown, and voluntarily emitted XML-like headings receive the same substantive treatment. Answer extraction remains separate from substantive trajectory scoring.

## Tests

Tests are grouped by the behavior they cover:

- answer and choice normalization: `test_answer_utils.py`, `test_datasets.py`;
- protocol rendering, role recovery, and judgment semantics: `test_io_models_protocol.py`, `test_judge.py`;
- TF-IDF weighting, conservative matching, anti-spam behavior, and lexical reference freezing: `test_lexical.py`;
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

The project requires Python 3.12.13 or newer and the dependency ranges declared in `pyproject.toml`. There are no compatibility branches for older Python releases or dependency APIs. Route schema 2 remains readable only so a 3.0.1/3.0.2 workspace can resume without changing its already used provider; all newly pinned routes use concurrent-load-qualified schema 3.
