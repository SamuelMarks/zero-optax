"""zero_optax schedules package."""

from __future__ import annotations

from zero_optax.schedules import _schedule
from zero_optax.schedules.base import (
    constant_schedule,
    cosine_decay_schedule,
    exponential_decay,
    join_schedules,
    linear_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    polynomial_schedule,
)
from zero_optax.schedules.cycle import (
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
)
from zero_optax.schedules.inject import (
    InjectHyperparamsState,
    InjectStatefulHyperparamsState,
    inject_hyperparams,
    inject_stateful_hyperparams,
)
from zero_optax.schedules.schedule import (
    GrainDatasetOrSampler,
    grain_warmup_cosine_decay_schedule,
    steps_per_epoch_from_grain,
)
from zero_optax.schedules.warmup import (
    warmup_constant_schedule,
    warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule,
)

__all__ = [
    "GrainDatasetOrSampler",
    "InjectHyperparamsState",
    "InjectStatefulHyperparamsState",
    "_schedule",
    "constant_schedule",
    "cosine_decay_schedule",
    "cosine_onecycle_schedule",
    "exponential_decay",
    "grain_warmup_cosine_decay_schedule",
    "inject_hyperparams",
    "inject_stateful_hyperparams",
    "join_schedules",
    "linear_onecycle_schedule",
    "linear_schedule",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polynomial_schedule",
    "sgdr_schedule",
    "steps_per_epoch_from_grain",
    "warmup_constant_schedule",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
]
