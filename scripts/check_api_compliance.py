import optax
import zero_optax
import inspect


def get_public_api(module):
    public_api = set()
    for name in dir(module):
        if not name.startswith("_"):
            obj = getattr(module, name)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                public_api.add(name)
    return public_api


optax_api = get_public_api(optax)
zero_optax_api = get_public_api(zero_optax)

for sub in ["losses", "schedules"]:
    if hasattr(optax, sub) and hasattr(zero_optax, sub):
        optax_api.update(
            f"{sub}.{name}" for name in get_public_api(getattr(optax, sub))
        )
        zero_optax_api.update(
            f"{sub}.{name}" for name in get_public_api(getattr(zero_optax, sub))
        )

implemented = zero_optax_api.intersection(optax_api)
total_optax = len(optax_api)
missing = optax_api - zero_optax_api

print(f"Total optax API items: {total_optax}")
print(f"Total zero_optax implemented items: {len(implemented)}")
print(f"Compliance: {len(implemented) / total_optax * 100:.2f}%")
