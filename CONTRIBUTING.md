# Contributing

Contributions are welcome when they preserve the experiment's reproducibility and make the implementation easier to understand or maintain.

## Set up a development environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

Use Python 3.12.13 or newer. The exact direct dependencies used for research runs and development checks are recorded in the two lock files at the repository root.

## Before changing experimental behavior

Read these files first:

- `docs/experiment.md` for the experimental design and reported outcomes;
- `docs/development.md` for module ownership and repository checks;
- `src/instruction_duplication/protocol.py` for the authoritative protocol and conditions.

Changes to any of the following affect the identity or interpretation of a run and must be treated deliberately:

- dataset repositories, revisions, splits, or normalization rules;
- model identifiers, provider routes, prices, or output limits;
- prompt conditions or protocol text;
- parser, judge, lexical scorer, or statistical families;
- workspace files, hashes, or database semantics.

Update the corresponding manifest inputs and regression tests whenever one of these changes.

## Coding expectations

Keep untyped external data at explicit validation boundaries. Internal records should remain concrete, immutable where practical, and strictly typed. The repository checks reject `Any`, unchecked casts, type-checker suppressions, and direct printing outside the CLI.

Do not add silent provider or dataset fallbacks, best-effort XML repair, unversioned judgments, or concurrent database writes from request workers. Failures and retries must remain visible in attempt-level provenance.

A bug found in a real run should first be reduced to a deterministic, network-free regression test. Tests should describe the behavior being protected rather than the implementation detail used to achieve it.

## Run the checks

```bash
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

## Pull requests

Keep each change focused. The pull-request description should explain:

- what changed and why;
- whether experiment semantics or workspace compatibility changed;
- which checks were run;
- whether model limits, provider prices, or dataset inputs changed.
