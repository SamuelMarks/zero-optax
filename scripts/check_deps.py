import ast
import sys
from pathlib import Path

# Allowlist of acceptable modules
ALLOWED_MODULES = {
    "numpy",
    "pydantic",
    "cdd_python",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_jax",
    "ml_switcheroo",  # allowed synonym for ml_switcheroo_compiler components
    "chex",
}

# Standard library modules are intrinsically allowed.
# This is a naive heuristic: if it's not in sys.stdlib_module_names,
# we check our whitelist.
try:
    STDLIB = set(sys.stdlib_module_names)
except AttributeError:
    # Fallback for Python < 3.10
    from stdlib_list import stdlib_list

    STDLIB = set(stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))


def get_base_module(module_name):
    """Return the base module name (e.g., 'zero_jax.numpy' -> 'zero_jax')."""
    return module_name.split(".")[0]


def check_file(filepath):
    """Check a Python file for disallowed imports."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(filepath))
        except SyntaxError:
            return []  # Let Ruff/linters handle syntax errors

    violations = []

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
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = get_base_module(node.module)
                if (
                    base_module not in STDLIB
                    and base_module not in ALLOWED_MODULES
                    and not base_module.startswith("zero_optax")
                ):
                    violations.append((node.lineno, node.module))

    return violations


def main():
    src_dir = Path("src")
    if not src_dir.exists():
        return 0

    all_violations = []

    for py_file in src_dir.rglob("*.py"):
        # Ignore tests if they somehow ended up in src, though they shouldn't be
        if "tests" in py_file.parts:
            continue

        violations = check_file(py_file)
        for lineno, module in violations:
            all_violations.append(
                f"{py_file}:{lineno}: Disallowed 3rd-party import '{module}'."
            )

    if all_violations:
        print("Disallowed 3rd-party dependencies found in non-test code:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nAllowed 3rd-party modules: numpy, pydantic, cdd_python, ml_switcheroo_ir, ml_switcheroo_compiler, zero_jax"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
