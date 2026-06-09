import ast
import sys
from pathlib import Path


def check_file(path):
    with open(path) as f:
        tree = ast.parse(f.read())

    missing = []

    # Module docstring
    if not ast.get_docstring(tree):
        missing.append(f"{path}: Module docstring missing")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                missing.append(
                    f"{path}: {node.__class__.__name__} '{node.name}' docstring missing"
                )

    return missing


paths = list(Path("src/zero_optax").rglob("*.py"))
all_missing = []
for p in paths:
    all_missing.extend(check_file(p))

if all_missing:
    for m in all_missing:
        print(m)
    sys.exit(1)
print("All docs present!")
