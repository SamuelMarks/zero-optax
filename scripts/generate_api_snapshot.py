"""Generate the `tests/optax_api_snapshot.json` file from the installed `optax` package."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import ModuleType

import optax


def get_public_api(
    module: ModuleType,
) -> dict[str, list[dict[str, str | int | float | bool | None]]]:
    """Extract parameter signature metadata for all public callables in `module`.

    Args:
        module: Python module to introspect.

    Returns:
        Mapping from function/class name to list of parameter descriptor dictionaries.
    """
    public_api: dict[str, list[dict[str, str | int | float | bool | None]]] = {}
    mod_vars = dict(inspect.getmembers(module))
    raw_all = mod_vars.get("__all__")
    names: list[str] = (
        [str(n) for n in raw_all]
        if isinstance(raw_all, (list, tuple))
        else [n for n in mod_vars if not n.startswith("_")]
    )
    for name in names:
        if name not in mod_vars:
            continue
        obj = mod_vars[name]
        if inspect.isfunction(obj) or inspect.isclass(obj):
            sig = inspect.signature(obj)
            params: list[dict[str, str | int | float | bool | None]] = []
            for p_name, p in sig.parameters.items():
                has_default = p.default is not inspect.Parameter.empty
                default_val: str | int | float | bool | None = None
                if has_default:
                    if (
                        isinstance(p.default, (int, float, str, bool))
                        or p.default is None
                    ):
                        default_val = p.default
                    else:
                        default_val = "<non-primitive>"
                params.append(
                    {
                        "name": p_name,
                        "kind": str(p.kind),
                        "has_default": has_default,
                        "default": default_val,
                    }
                )
            public_api[name] = params
    return public_api


def generate_snapshot(
    output_path: Path | None = None,
) -> dict[str, dict[str, list[dict[str, str | int | float | bool | None]]]]:
    """Build the API snapshot dictionary for `optax`, `optax.losses`, and `optax.schedules`.

    Args:
        output_path: Optional path to write JSON snapshot (defaults to `tests/optax_api_snapshot.json`).

    Returns:
        Complete snapshot dictionary.
    """
    target_path = (
        Path("tests/optax_api_snapshot.json") if output_path is None else output_path
    )
    api: dict[str, dict[str, list[dict[str, str | int | float | bool | None]]]] = {
        "root": get_public_api(optax)
    }
    optax_vars = dict(inspect.getmembers(optax))
    for sub in ("losses", "schedules", "optimizers", "combine"):
        sub_mod = optax_vars.get(sub)
        if isinstance(sub_mod, ModuleType):
            api[sub] = get_public_api(sub_mod)

    target_path.write_text(
        json.dumps(api, indent=2, sort_keys=True) + chr(10), encoding="utf-8"
    )
    return api


if __name__ == "__main__":
    generate_snapshot()
