"""zero_optax schedules package."""

from zero_optax.schedules.base import (
    constant_schedule,
    linear_schedule,
    polynomial_schedule,
    exponential_decay,
    cosine_decay_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    join_schedules,
)
from zero_optax.schedules.warmup import (
    warmup_constant_schedule,
    warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule,
)
from zero_optax.schedules.cycle import (
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
)
from zero_optax.schedules.inject import (
    inject_hyperparams,
    inject_stateful_hyperparams,
)

__all__ = [
    "constant_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "exponential_decay",
    "cosine_decay_schedule",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "join_schedules",
    "warmup_constant_schedule",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
    "cosine_onecycle_schedule",
    "linear_onecycle_schedule",
    "sgdr_schedule",
    "inject_hyperparams",
    "inject_stateful_hyperparams",
]
