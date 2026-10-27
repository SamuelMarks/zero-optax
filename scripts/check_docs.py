"""CLI entry point for verifying 100% Google-style docstring coverage across `src/zero_optax`, `scripts`, and `tests`."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lint_typing_and_docs import main as _lint_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AST docstring and typing linter across target directories.

    Args:
        argv: Optional sequence of target directory or file paths.

    Returns:
        Exit code `0` if 100% compliant, or `1` if any docstring or typing violations are found.
    """
    return _lint_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
