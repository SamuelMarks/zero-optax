"""Hyperparameter injection schedules and states."""

from __future__ import annotations

from zero_optax.schedules.schedule import (
    InjectHyperparamsState,
    InjectStatefulHyperparamsState,
    inject_hyperparams,
    inject_stateful_hyperparams,
)

__all__ = [
    "InjectHyperparamsState",
    "InjectStatefulHyperparamsState",
    "inject_hyperparams",
    "inject_stateful_hyperparams",
]
