"""CLI entry point for verifying zero `Any`, `object`, `cast`, or dynamic attribute reflection."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lint_typing_and_docs import main as _lint_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AST strict typing linter across target directories.

    Args:
        argv: Optional sequence of target directory or file paths.

    Returns:
        Exit code `0` if 100% compliant, or `1` if any typing violations are found.
    """
    return _lint_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
