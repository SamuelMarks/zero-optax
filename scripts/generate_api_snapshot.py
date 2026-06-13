import optax
import inspect
import json


def get_public_api(module):
    public_api = {}
    names = getattr(
        module, "__all__", [n for n in dir(module) if not n.startswith("_")]
    )
    for name in names:
        obj = getattr(module, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                sig = inspect.signature(obj)
                params = []
                for pname, p in sig.parameters.items():
                    has_default = p.default is not inspect.Parameter.empty
                    default_val = None
                    if has_default:
                        if isinstance(p.default, (int, float, str, bool, type(None))):
                            default_val = p.default
                        else:
                            default_val = "<non-primitive>"
                    params.append(
                        {
                            "name": pname,
                            "kind": str(p.kind),
                            "has_default": has_default,
                            "default": default_val,
                        }
                    )
                public_api[name] = params
            except ValueError:
                pass
    return public_api


api = {}
api["root"] = get_public_api(optax)
# Also snapshot submodules if needed, though they mirror root.
for sub in [
    "losses",
    "schedules",
]:  # optimizers and combine are not standard modules in optax usually
    if hasattr(optax, sub):
        api[sub] = get_public_api(getattr(optax, sub))

with open("tests/optax_api_snapshot.json", "w") as f:
    json.dump(api, f, indent=2, sort_keys=True)
