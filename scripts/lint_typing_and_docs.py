"""AST-based static linter enforcing 100% docstrings and strict typing invariants.

Checks across `src/`, `scripts/`, and `tests/`:
1. Module, class, and function docstrings must be present.
2. Every function parameter (except `self` and `cls`) must be documented in `Args:`.
3. Every function parameter and return type must have an explicit type annotation.
4. No usage of `Any`, `object`, `typing.cast`, `vars`, or `__dict__`.
5. No usage of `getattr`, `setattr`, or `hasattr` in `src/`.
6. No `# type: ignore` comments anywhere in `src/`, `scripts/`, or `tests/`.
"""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

FORBIDDEN_NAMES: set[str] = {"Any", "object", "cast", "vars"}
FORBIDDEN_ATTRS: set[str] = {"__dict__"}
FORBIDDEN_REFLECTION_IN_SRC: set[str] = {"getattr", "setattr", "hasattr"}


def _extract_documented_args(docstring: str) -> set[str]:
    """Parse parameter names documented in a Google-style `Args:` section.

    Args:
        docstring: Raw docstring text of a function or method.

    Returns:
        Set of parameter names (without leading `*` or `**`) found in `Args:`.
    """
    documented: set[str] = set()
    in_args = False
    args_indent: int | None = None

    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if re.match(r"^Args:\s*$", stripped):
            in_args = True
            args_indent = indent
            continue
        if in_args:
            if (
                args_indent is not None
                and indent <= args_indent
                and stripped.endswith(":")
            ):
                in_args = False
                continue
            m = re.match(
                r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped
            )
            if m:
                documented.add(m.group(1))
    return documented


def lint_file(filepath: Path, is_src: bool = False) -> list[str]:
    """Lint a single Python file for typing, reflection, and docstring violations.

    Args:
        filepath: Path to the Python file to inspect.
        is_src: Whether the file resides under `src/` (enables reflection checks).

    Returns:
        List of formatted error messages for `filepath`.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    errors: list[str] = []

    with filepath.open("rb") as f_bytes:
        for tok in tokenize.tokenize(f_bytes.readline):
            if tok.type == tokenize.COMMENT and re.search(
                r"#\s*type:\s*ignore", tok.string
            ):
                errors.append(
                    f"{filepath}:{tok.start[0]}: Forbidden '# type: ignore' comment."
                )

    if not ast.get_docstring(tree):
        errors.append(f"{filepath}:1: Missing module docstring.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                errors.append(
                    f"{filepath}:{node.lineno}: Forbidden identifier '{node.id}'."
                )
            if is_src and node.id in FORBIDDEN_REFLECTION_IN_SRC:
                errors.append(
                    f"{filepath}:{node.lineno}: Forbidden reflection call '{node.id}' in src/."
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                errors.append(
                    f"{filepath}:{node.lineno}: Forbidden dunder attribute '{node.attr}'."
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    errors.append(
                        f"{filepath}:{node.lineno}: Forbidden import '{alias.name}'."
                    )
        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                errors.append(
                    f"{filepath}:{node.lineno}: Class '{node.name}' is missing a docstring."
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if not doc:
                errors.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' is missing a docstring."
                )
            if node.returns is None:
                errors.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' is missing a return type annotation."
                )

            all_args = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                all_args.append(node.args.vararg)
            if node.args.kwarg is not None:
                all_args.append(node.args.kwarg)

            non_self_args = [a for a in all_args if a.arg not in ("self", "cls")]
            for arg in non_self_args:
                if arg.annotation is None:
                    errors.append(
                        f"{filepath}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' is missing a type annotation."
                    )

            if doc and non_self_args:
                doc_args = _extract_documented_args(doc)
                for arg in non_self_args:
                    if arg.arg not in doc_args:
                        errors.append(
                            f"{filepath}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' is missing from docstring Args: section."
                        )

    return errors


def lint_paths(paths: Sequence[Path]) -> list[str]:
    """Lint a sequence of files or directories.

    Args:
        paths: Sequence of file or directory `Path` objects to check.

    Returns:
        Aggregated list of error strings across all scanned Python files.
    """
    all_errors: list[str] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            is_src = "src" in p.parts
            all_errors.extend(lint_file(p, is_src=is_src))
        else:
            is_src_dir = "src" in p.parts or p.name == "src"
            for py_file in sorted(p.rglob("*.py")):
                all_errors.extend(lint_file(py_file, is_src=is_src_dir))
    return all_errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the typing and docstring linter from CLI arguments.

    Args:
        argv: Optional list of directory/file path strings (defaults to `["src", "scripts", "tests"]`).

    Returns:
        `0` if all checks pass, `1` if any violation is detected.
    """
    target_strs = list(argv) if argv else ["src", "scripts", "tests"]
    target_paths = [Path(s) for s in target_strs if Path(s).exists()]
    errors = lint_paths(target_paths)
    if errors:
        for err in errors:
            print(err)
        return 1
    print("Typing and docstring lint passed (100% compliant).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
