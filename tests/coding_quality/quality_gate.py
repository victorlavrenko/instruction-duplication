"""Small dependency-free repository checks that complement Ruff and type checkers."""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "instruction_duplication"
PYTHON_ROOTS = (SOURCE, ROOT / "tests", ROOT / "tools")
FORBIDDEN_REPOSITORY_PATHS = (
    ROOT / "PROTOCOL.txt",
    ROOT / "setup.cfg",
    ROOT / "PKG-INFO",
    ROOT / "src" / "instruction_duplication.egg-info",
)
FORBIDDEN_STORAGE_APIS = (
    "def record_attempt(",
    "def finish_cell(",
    "def mark_budget_blocked(",
)
REQUIRED_WORKFLOW_COMMANDS = (
    "python -m ruff check .",
    "python -m ruff format --check .",
    "python -m pyright",
    "python -m mypy",
)
FORBIDDEN_SOURCE_TEXT = (
    "TypeAlias",
    "timezone.utc",
    "dt.timezone.utc",
)


def _location(path: Path, line: int) -> str:
    return f"{path.relative_to(ROOT)}:{line}"


def _comment_errors(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        comment = token.string.casefold()
        if "type: ignore" in comment:
            errors.append(f"{_location(path, token.start[0])}: type-ignore is forbidden")
        if re.search(r"#\s*noqa\b", comment):
            errors.append(f"{_location(path, token.start[0])}: noqa is forbidden")
    return errors


def _annotation_errors(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        location = _location(path, getattr(node, "lineno", 1))
        if isinstance(node, ast.Name) and node.id == "Any":
            errors.append(f"{location}: typing.Any is forbidden")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cast"
        ):
            errors.append(f"{location}: typing.cast is forbidden")
    return errors


def _print_errors(path: Path, tree: ast.AST) -> list[str]:
    if path.parent != SOURCE or path.name == "cli.py":
        return []
    return [
        f"{_location(path, node.lineno)}: print outside CLI"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def inspect_file(path: Path) -> list[str]:
    """Inspect one Python file for local repository invariants."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{_location(path, exc.lineno or 1)}: syntax error: {exc.msg}"]
    deprecated = [
        f"{path.relative_to(ROOT)}: deprecated or pre-3.12 construct remains: {value}"
        for value in FORBIDDEN_SOURCE_TEXT
        if path.parent == SOURCE and value in text
    ]
    return [
        *_comment_errors(path, text),
        *_annotation_errors(path, tree),
        *_print_errors(path, tree),
        *deprecated,
    ]


def inspect_configuration() -> list[str]:
    """Check that the CI workflow invokes every configured static-analysis tool."""
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    return [
        f"quality.yml: missing static check: {command}"
        for command in REQUIRED_WORKFLOW_COMMANDS
        if command not in workflow
    ]


def inspect_repository() -> list[str]:
    """Check repository hygiene and removed APIs."""
    errors = [
        f"{path.relative_to(ROOT)}: file should not be committed"
        for path in FORBIDDEN_REPOSITORY_PATHS
        if path.exists()
    ]
    storage_text = (SOURCE / "storage.py").read_text(encoding="utf-8")
    errors.extend(
        f"storage.py: removed API remains: {name}"
        for name in FORBIDDEN_STORAGE_APIS
        if name in storage_text
    )
    return errors


def _python_files() -> list[Path]:
    return [path for root in PYTHON_ROOTS for path in sorted(root.rglob("*.py"))]


def main() -> int:
    """Run all dependency-free repository checks."""
    errors = [error for path in _python_files() for error in inspect_file(path)]
    errors.extend(inspect_configuration())
    errors.extend(inspect_repository())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("quality_gate=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
