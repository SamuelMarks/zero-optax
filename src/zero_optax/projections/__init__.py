"""Euclidean and constrained projections (`zero_optax.projections`)."""

from __future__ import annotations

from zero_optax.projections._projections import (
    projection_box,
    projection_halfspace,
    projection_hypercube,
    projection_hyperplane,
    projection_l1_ball,
    projection_l1_sphere,
    projection_l2_ball,
    projection_l2_sphere,
    projection_linf_ball,
    projection_non_negative,
    projection_simplex,
    projection_vector,
)

__all__ = [
    "projection_box",
    "projection_halfspace",
    "projection_hypercube",
    "projection_hyperplane",
    "projection_l1_ball",
    "projection_l1_sphere",
    "projection_l2_ball",
    "projection_l2_sphere",
    "projection_linf_ball",
    "projection_non_negative",
    "projection_simplex",
    "projection_vector",
]
