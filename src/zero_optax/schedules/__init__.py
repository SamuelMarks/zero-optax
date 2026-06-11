"""zero_optax schedules package."""

import ml_switcheroo

from zero_optax.schedules.schedule import (
    constant_schedule,
    join_schedules,
    warmup_constant_schedule,
    warmup_exponential_decay_schedule,
    exponential_decay,
    cosine_decay_schedule,
    warmup_cosine_decay_schedule,
    cosine_onecycle_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    linear_schedule,
    polynomial_schedule,
    linear_onecycle_schedule,
    inject_hyperparams,
    sgdr_schedule,
    inject_stateful_hyperparams,
)

__all__ = [
    "constant_schedule",
    "join_schedules",
    "warmup_constant_schedule",
    "warmup_exponential_decay_schedule",
    "exponential_decay",
    "cosine_decay_schedule",
    "warmup_cosine_decay_schedule",
    "cosine_onecycle_schedule",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "linear_schedule",
    "polynomial_schedule",
    "linear_onecycle_schedule",
    "inject_hyperparams",
    "sgdr_schedule",
    "inject_stateful_hyperparams",
]
