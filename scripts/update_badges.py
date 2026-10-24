"""Update test coverage, documentation coverage, and API compliance badges in `README.md`."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path


def get_color(pct: float) -> str:
    """Determine the shields.io badge color name for a given percentage.

    Args:
        pct: Coverage or compliance percentage in `[0, 100]`.

    Returns:
        Shields.io color identifier string.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage without trailing `.0` when integer-valued.

    Args:
        cov: Coverage percentage float.

    Returns:
        Formatted percentage string.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage(cov_json_path: Path | None = None, run_cmd: bool = True) -> float:
    """Read test coverage percentage from `coverage.json`.

    Args:
        cov_json_path: Optional path to `coverage.json`.
        run_cmd: Whether to invoke `coverage json -o coverage.json` before reading.

    Returns:
        Test coverage percentage in `[0.0, 100.0]`.
    """
    target_path = Path("coverage.json") if cov_json_path is None else cov_json_path
    if run_cmd:
        subprocess.run(
            ["coverage", "json", "-o", str(target_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not target_path.exists():
        return 100.0
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        totals = data.get("totals", {})
        pct = totals.get("percent_covered", 100.0)
        return float(pct)
    except (ValueError, KeyError, OSError):
        return 0.0


def get_doc_coverage(src_dir: Path | None = None) -> float:
    """Calculate AST docstring coverage across `src_dir`.

    Args:
        src_dir: Directory to inspect (defaults to `Path("src/zero_optax")`).

    Returns:
        Docstring coverage percentage in `[0.0, 100.0]`.
    """
    target_dir = Path("src/zero_optax") if src_dir is None else src_dir
    if not target_dir.exists():
        return 100.0
    total = 0
    documented = 0
    for py_file in sorted(target_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        total += 1
        if ast.get_docstring(tree):
            documented += 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node):
                    documented += 1
    if total == 0:
        return 100.0
    return (documented / total) * 100.0


def get_api_compliance(root_dir: Path | None = None) -> float:
    """Read overall API compliance percentage from `*_TODO.md`.

    Args:
        root_dir: Directory containing `OPTAX_TODO.md` (defaults to current working directory).

    Returns:
        API compliance percentage in `[0.0, 100.0]`.
    """
    base_dir = Path(".") if root_dir is None else root_dir
    todo_files = sorted(f for f in os.listdir(base_dir) if f.endswith("_TODO.md"))
    if not todo_files:
        return 0.0
    todo_path = base_dir / todo_files[0]
    for line in todo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Overall Compliance:"):
            raw_val = line.split(":", 1)[1].strip().replace("%", "")
            try:
                return float(raw_val)
            except ValueError:
                return 0.0
    return 0.0


def update_readme(
    readme_path: Path | None = None,
    cov_json_path: Path | None = None,
    src_dir: Path | None = None,
    root_dir: Path | None = None,
    run_cmd: bool = True,
) -> None:
    """Update shield badges in `README.md`.

    Args:
        readme_path: Path to `README.md` (defaults to `Path("README.md")`).
        cov_json_path: Optional path to `coverage.json`.
        src_dir: Optional path to source directory for docstring coverage.
        root_dir: Optional root directory containing `*_TODO.md`.
        run_cmd: Whether to invoke `coverage json` subprocess.
    """
    target_readme = Path("README.md") if readme_path is None else readme_path
    if not target_readme.exists():
        return

    test_cov = get_test_coverage(cov_json_path=cov_json_path, run_cmd=run_cmd)
    doc_cov = get_doc_coverage(src_dir=src_dir)
    api_comp = get_api_compliance(root_dir=root_dir)

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)
    api_str = format_cov(api_comp)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)
    api_color = get_color(api_comp)

    content = target_readme.read_text(encoding="utf-8")

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    api_re = re.compile(
        r"\[?\!\[API Compliance\]\(https://img\.shields\.io/badge/(?:[aA]pi_)?(?:[cC]ompliance)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    if api_re.search(content):
        content = api_re.sub(
            f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)",
            content,
        )
    else:
        doc_badge_match = re.search(r"\[\!\[Doc Coverage\][^\r\n]*", content)
        if doc_badge_match:
            insert_pos = doc_badge_match.end()
            content = (
                content[:insert_pos]
                + chr(10)
                + f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)"
                + content[insert_pos:]
            )

    target_readme.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    update_readme()
