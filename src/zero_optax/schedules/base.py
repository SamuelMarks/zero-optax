"""Module docstring."""

"""Base schedules."""
from zero_optax.schedules.schedule import (
    constant_schedule,
    linear_schedule,
    polynomial_schedule,
    exponential_decay,
    cosine_decay_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    join_schedules,
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
]
