import json
import inspect
import pytest
from pathlib import Path
import zero_optax

with open(Path(__file__).parent / "optax_api_snapshot.json", "r") as f:
    SNAPSHOT = json.load(f)["root"]


def get_exported_functions(module):
    funcs = {}
    names = getattr(
        module, "__all__", [n for n in dir(module) if not n.startswith("_")]
    )
    for name in names:
        obj = getattr(module, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            funcs[name] = obj
    return funcs


def test_api_parity():
    z_funcs = get_exported_functions(zero_optax)
    mismatches = []

    for fname, z_obj in z_funcs.items():
        if fname not in SNAPSHOT:
            continue

        optax_params = SNAPSHOT[fname]
        try:
            z_sig = inspect.signature(z_obj)
        except ValueError:
            continue

        z_params_list = []
        for pname, p in z_sig.parameters.items():
            has_default = p.default is not inspect.Parameter.empty
            default_val = None
            if has_default:
                if isinstance(p.default, (int, float, str, bool, type(None))):
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
            ):
                if zp["default"] != op["default"]:
                    mismatches.append(
                        f"{fname}: default val mismatch for {zp['name']} ({zp['default']} != {op['default']})"
                    )
                    continue

    if mismatches:
        pytest.fail("API Parity Mismatches found:\n" + "\n".join(mismatches))
