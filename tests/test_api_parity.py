"""API signature parity tests against `optax_api_snapshot.json` and `optax_v0.2.8.json`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import FunctionType, ModuleType

import pytest

import zero_optax
from scripts.check_compliance import verify_snapshot_v028
from zero_optax import combine

SNAPSHOT_PATH = Path(__file__).parent / "optax_api_snapshot.json"
with SNAPSHOT_PATH.open("r", encoding="utf-8") as _f:
    FULL_SNAPSHOT: dict[
        str, dict[str, list[dict[str, str | int | float | bool | None]]]
    ] = json.load(_f)
SNAPSHOT = FULL_SNAPSHOT["root"]


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


def _compare_module_against_snapshot(
    module: ModuleType,
    section_snap: dict[str, list[dict[str, str | int | float | bool | None]]],
) -> list[str]:
    """Compare all exported functions in `module` against `section_snap`.

    Args:
        module: Module whose exported functions are inspected.
        section_snap: Snapshot dictionary mapping function names to parameter lists.

    Returns:
        List of mismatch descriptions.
    """
    z_funcs = get_exported_functions(module)
    mismatches: list[str] = []

    for fname, z_obj in z_funcs.items():
        if fname not in section_snap or inspect.isclass(z_obj):
            continue

        optax_params = section_snap[fname]
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
    return mismatches


def test_api_parity_v028_snapshot() -> None:
    """Assert 100% signature parity against `ml-ecosystem-snapshots` `optax_v0.2.8.json`."""
    errors = verify_snapshot_v028()
    assert errors == [], "Snapshot v0.2.8 mismatches: " + "; ".join(errors)


def test_api_parity_local_snapshot() -> None:
    """Assert parameter names, defaults, submodules, and `*State` classes match `tests/optax_api_snapshot.json`."""
    submodules: dict[str, ModuleType] = {
        "root": zero_optax,
        "losses": zero_optax.losses,
        "schedules": zero_optax.schedules,
        "optimizers": zero_optax.optimizers,
        "combine": combine,
    }
    mismatches: list[str] = []
    for section_name, mod in submodules.items():
        if section_name in FULL_SNAPSHOT:
            mismatches.extend(
                _compare_module_against_snapshot(mod, FULL_SNAPSHOT[section_name])
            )

    root_vars = dict(inspect.getmembers(zero_optax))
    expected_states = {
        name
        for name in zero_optax.__all__
        if name.endswith("State") and name != "OptState"
    }
    verified_states: set[str] = set()
    for name in sorted(expected_states):
        assert name in root_vars, f"Missing exported state class {name}"
        cls_obj = root_vars[name]
        assert inspect.isclass(cls_obj), f"{name} should be a class"
        cls_members = dict(inspect.getmembers(cls_obj))
        assert "_fields" in cls_members, f"{name} must define _fields"
        assert "_asdict" in cls_members, f"{name} must define _asdict"
        assert "_make" in cls_members, f"{name} must define _make"
        verified_states.add(name)
    assert verified_states == expected_states
    assert len(verified_states) == 52

    if mismatches:
        pytest.fail("API Parity Mismatches found: " + "; ".join(mismatches))
