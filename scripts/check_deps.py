"""Check `src/zero_optax` for disallowed third-party runtime imports."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED_MODULES: set[str] = {
    "pydantic",
    "cdd",
    "cdd_python",
    "ml_ecosystem_snapshots",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "ml_switcheroo",
    "zero_jax",
    "zero_chex",
    "zero_flax",
    "zero_orbax",
    "zero_grain",
}


FALLBACK_STDLIB_MODULES: set[str] = {
    "__future__",
    "abc",
    "ast",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "enum",
    "functools",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "re",
    "shutil",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "time",
    "tokenize",
    "types",
    "typing",
    "typing_extensions",
    "unittest",
    "warnings",
}


def _get_stdlib_modules(custom_names: set[str] | None = None) -> set[str]:
    """Return the set of standard library top-level module names.

    Args:
        custom_names: Optional explicit set of module names for testing.

    Returns:
        Set of standard library module names.
    """
    if custom_names is not None:
        return {str(n) for n in custom_names}
    return FALLBACK_STDLIB_MODULES | set(sys.builtin_module_names)


STDLIB: set[str] = _get_stdlib_modules()


def get_base_module(module_name: str) -> str:
    """Return the root package name from a dotted module path.

    Args:
        module_name: Dotted Python import path (e.g., `'ml_switcheroo_compiler.ops'`).

    Returns:
        Top-level package name (e.g., `'ml_switcheroo_compiler'`).
    """
    return module_name.split(".")[0]


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Scan a single Python source file for disallowed third-party imports.

    Args:
        filepath: Path to the Python file to inspect.

    Returns:
        List of `(line_number, module_name)` tuples for each disallowed import
        (or `(lineno, "<SyntaxError>")` if `filepath` cannot be parsed).
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        return [(exc.lineno or 1, "<SyntaxError>")]

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = get_base_module(alias.name)
                if (
                    base_module not in STDLIB
                    and base_module not in ALLOWED_MODULES
                    and not base_module.startswith("zero_optax")
                ):
                    violations.append((node.lineno, alias.name))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.level == 0
        ):
            base_module = get_base_module(node.module)
            if (
                base_module not in STDLIB
                and base_module not in ALLOWED_MODULES
                and not base_module.startswith("zero_optax")
            ):
                violations.append((node.lineno, node.module))

    return violations


def check_directory(src_dir: Path | None = None) -> list[str]:
    """Check all Python files in `src_dir` for disallowed imports.

    Args:
        src_dir: Root source directory to scan (defaults to `Path("src")`).

    Returns:
        Formatted violation strings across all scanned files.
    """
    target_dir = Path("src") if src_dir is None else src_dir
    if not target_dir.exists():
        return []

    all_violations: list[str] = []
    for py_file in sorted(target_dir.rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        for lineno, module in check_file(py_file):
            all_violations.append(
                f"{py_file}:{lineno}: Disallowed 3rd-party import '{module}'."
            )
    return all_violations


def main(src_dir: Path | None = None) -> int:
    """Run the third-party import check and return an exit code.

    Args:
        src_dir: Optional source directory override for testing.

    Returns:
        `0` if no disallowed imports are found, `1` otherwise.
    """
    all_violations = check_directory(src_dir)
    if all_violations:
        print("Disallowed 3rd-party dependencies found in non-test code:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "Allowed 3rd-party modules: pydantic, cdd_python, ml_switcheroo_ir, ml_switcheroo_compiler"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
