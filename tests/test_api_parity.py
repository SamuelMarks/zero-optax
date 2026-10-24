"""API signature parity tests against `optax_api_snapshot.json` and `optax_v0.2.8.json`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import FunctionType, ModuleType

import pytest

import zero_optax
from scripts.check_compliance import verify_snapshot_v028

SNAPSHOT_PATH = Path(__file__).parent / "optax_api_snapshot.json"
with SNAPSHOT_PATH.open("r", encoding="utf-8") as _f:
    SNAPSHOT: dict[str, list[dict[str, str | int | float | bool | None]]] = json.load(
        _f
    )["root"]


def get_exported_functions(module: ModuleType) -> dict[str, FunctionType]:
    """Collect all public functions listed in `module.__all__`.

    Args:
        module: Target Python module (`zero_optax`).

    Returns:
        Dictionary mapping exported symbol names to their function objects.
    """
    funcs: dict[str, FunctionType] = {}
    mod_vars = dict(inspect.getmembers(module))
    raw_all = mod_vars.get("__all__", [])
    names = [str(n) for n in raw_all] if isinstance(raw_all, list) else []
    for name in names:
        if name in mod_vars:
            obj = mod_vars[name]
            if isinstance(obj, FunctionType):
                funcs[name] = obj
    return funcs


def test_api_parity_v028_snapshot() -> None:
    """Assert 100% signature parity against `ml-ecosystem-snapshots` `optax_v0.2.8.json`."""
    errors = verify_snapshot_v028()
    assert errors == [], "Snapshot v0.2.8 mismatches: " + "; ".join(errors)


def test_api_parity_local_snapshot() -> None:
    """Assert parameter names and defaults match `tests/optax_api_snapshot.json`."""
    z_funcs = get_exported_functions(zero_optax)
    mismatches: list[str] = []

    for fname, z_obj in z_funcs.items():
        if fname not in SNAPSHOT or inspect.isclass(z_obj):
            continue

        optax_params = SNAPSHOT[fname]
        z_sig = inspect.signature(z_obj)

        z_params_list: list[dict[str, str | int | float | bool | None]] = []
        for pname, p in z_sig.parameters.items():
            has_default = p.default is not inspect.Parameter.empty
            default_val: str | int | float | bool | None = None
            if has_default:
                if isinstance(p.default, (int, float, str, bool)) or p.default is None:
                    default_val = p.default
                else:
                    default_val = "<non-primitive>"
            z_params_list.append(
                {
                    "name": pname,
                    "kind": str(p.kind),
                    "has_default": has_default,
                    "default": default_val,
                }
            )

        if len(z_params_list) != len(optax_params):
            mismatches.append(
                f"{fname}: len mismatch ({len(z_params_list)} vs {len(optax_params)})"
            )
            continue

        for zp, op in zip(z_params_list, optax_params):
            if zp["name"] != op["name"]:
                mismatches.append(
                    f"{fname}: param name mismatch {zp['name']} != {op['name']}"
                )
                continue
            if zp["has_default"] != op["has_default"]:
                mismatches.append(f"{fname}: default mismatch for {zp['name']}")
                continue
            if (
                zp["has_default"]
                and zp["default"] != "<non-primitive>"
                and op["default"] != "<non-primitive>"
                and zp["default"] != op["default"]
            ):
                mismatches.append(
                    f"{fname}: default val mismatch for {zp['name']} ({zp['default']} != {op['default']})"
                )

    if mismatches:
        pytest.fail("API Parity Mismatches found: " + "; ".join(mismatches))
