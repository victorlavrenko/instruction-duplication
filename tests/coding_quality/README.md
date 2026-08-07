# Coding-quality checks

This directory contains repository-level tests that do not belong to a single runtime module.

`quality_gate.py` checks source and repository conventions without importing optional development tools. `test_configuration.py` verifies that the supported Python version, strict type checking, async test dependency, and CI commands remain configured.

Run them with:

```bash
python tests/coding_quality/quality_gate.py
pytest tests/coding_quality
```
