"""Compare public callable API exports between `optax` and `zero_optax`."""

from __future__ import annotations

import inspect
from types import ModuleType

import optax

import zero_optax


def get_public_api(module: ModuleType) -> set[str]:
    """Return all public functions and classes exposed on `module`.

    Args:
        module: Imported Python module to inspect.

    Returns:
        Set of public function and class names defined on `module`.
    """
    public_api: set[str] = set()
    for name, obj in inspect.getmembers(module):
        if not name.startswith("_") and (
            inspect.isfunction(obj) or inspect.isclass(obj)
        ):
            public_api.add(name)
    return public_api


def compute_api_compliance(
    ref_module: ModuleType = optax,
    target_module: ModuleType = zero_optax,
) -> tuple[int, int, float, set[str]]:
    """Compute public API overlap between `ref_module` and `target_module`.

    Args:
        ref_module: Reference framework module (defaults to `optax`).
        target_module: Target framework module (defaults to `zero_optax`).

    Returns:
        Tuple of `(total_ref_count, implemented_count, compliance_pct, missing_names)`.
    """
    optax_api = get_public_api(ref_module)
    zero_optax_api = get_public_api(target_module)
    implemented = optax_api.intersection(zero_optax_api)
    total_optax = len(optax_api)
    missing = optax_api - zero_optax_api
    pct = (len(implemented) / total_optax * 100.0) if total_optax > 0 else 100.0
    return total_optax, len(implemented), pct, missing


def main() -> int:
    """Print API compliance summary to stdout.

    Returns:
        `0` on completion.
    """
    total_optax, implemented_count, pct, _ = compute_api_compliance()
    print(f"Total optax API items: {total_optax}")
    print(f"Total zero_optax implemented items: {implemented_count}")
    print(f"Compliance: {pct:.2f}%")
    return 0


if __name__ == "__main__":
    main()
