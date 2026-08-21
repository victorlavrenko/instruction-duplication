from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_static_analysis_commands_are_configured() -> None:
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    for command in (
        "python -m ruff check .",
        "python -m ruff format --check .",
        "python -m pyright",
        "python -m mypy",
    ):
        assert command in workflow


def test_supported_python_and_strict_type_checking_are_configured() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12.13"' in pyproject
    assert 'target-version = "py312"' in pyproject
    assert 'python_version = "3.12"' in pyproject
    assert 'pythonVersion = "3.12"' in pyproject
    assert "strict = true" in pyproject
    assert 'typeCheckingMode = "strict"' in pyproject
    assert 'exclude = ["build", "dist", "tests"]' in pyproject


def test_async_test_dependency_is_declared() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    assert "pytest-asyncio" in pyproject
    assert "requirements-dev.lock" in workflow
    assert "pytest-asyncio" in (ROOT / "requirements-dev.lock").read_text(encoding="utf-8")

