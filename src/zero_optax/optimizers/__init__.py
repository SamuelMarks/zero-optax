"""zero_optax optimizers package."""

from __future__ import annotations

from zero_optax.optimizers.adabelief import AdaBeliefState, adabelief
from zero_optax.optimizers.adagrad import AdagradState, adagrad
from zero_optax.optimizers.adam import AdamState, adam
from zero_optax.optimizers.adamw import AdamwState, adamw
from zero_optax.optimizers.fromage import FromageState, fromage
from zero_optax.optimizers.lamb import LambState, lamb
from zero_optax.optimizers.lars import LarsState, lars
from zero_optax.optimizers.lion import LionState, lion
from zero_optax.optimizers.novograd import NovogradState, novograd
from zero_optax.optimizers.rmsprop import RmsPropState, rmsprop
from zero_optax.optimizers.sgd import SgdState, sgd
from zero_optax.optimizers.yogi import YogiState, yogi

__all__ = [
    "AdaBeliefState",
    "AdagradState",
    "AdamState",
    "AdamwState",
    "FromageState",
    "LambState",
    "LarsState",
    "LionState",
    "NovogradState",
    "RmsPropState",
    "SgdState",
    "YogiState",
    "adabelief",
    "adagrad",
    "adam",
    "adamw",
    "fromage",
    "lamb",
    "lars",
    "lion",
    "novograd",
    "rmsprop",
    "sgd",
    "yogi",
]
