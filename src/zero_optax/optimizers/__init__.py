"""zero_optax optimizers package."""

from __future__ import annotations

from zero_optax.optimizers.adabelief import AdaBeliefState, adabelief
from zero_optax.optimizers.adafactor import (
    FactoredState,
    adafactor,
    scale_by_factored_rms,
)
from zero_optax.optimizers.adagrad import AdagradState, adagrad
from zero_optax.optimizers.adam import AdamState, adam
from zero_optax.optimizers.adamax import (
    ScaleByAdamaxState,
    adamax,
    scale_by_adamax,
)
from zero_optax.optimizers.adamaxw import adamaxw
from zero_optax.optimizers.adamw import AdamwState, adamw
from zero_optax.optimizers.amsgrad import amsgrad
from zero_optax.optimizers.fromage import FromageState, fromage
from zero_optax.optimizers.lamb import LambState, lamb
from zero_optax.optimizers.lars import LarsState, lars
from zero_optax.optimizers.lbfgs import (
    ScaleByLBFGSState,
    lbfgs,
    scale_by_lbfgs,
)
from zero_optax.optimizers.lion import LionState, lion
from zero_optax.optimizers.noisy_sgd import (
    ScaleBySignState,
    noisy_sgd,
    scale_by_sign,
    sign_sgd,
)
from zero_optax.optimizers.novograd import NovogradState, novograd
from zero_optax.optimizers.ogd import (
    ScaleByOptimisticGradientState,
    optimistic_gradient_descent,
    scale_by_optimistic_gradient,
)
from zero_optax.optimizers.polyak_sgd import polyak_sgd, scale_by_polyak
from zero_optax.optimizers.radam import radam, scale_by_radam
from zero_optax.optimizers.rmsprop import RmsPropState, rmsprop
from zero_optax.optimizers.rprop import (
    ScaleByRpropState,
    ScaleBySm3State,
    rprop,
    scale_by_rprop,
    scale_by_sm3,
    sm3,
)
from zero_optax.optimizers.sgd import SgdState, sgd
from zero_optax.optimizers.yogi import YogiState, yogi

__all__ = [
    "AdaBeliefState",
    "AdagradState",
    "AdamState",
    "AdamwState",
    "FactoredState",
    "FromageState",
    "LambState",
    "LarsState",
    "LionState",
    "NovogradState",
    "RmsPropState",
    "ScaleByAdamaxState",
    "ScaleByLBFGSState",
    "ScaleByOptimisticGradientState",
    "ScaleByRpropState",
    "ScaleBySignState",
    "ScaleBySm3State",
    "SgdState",
    "YogiState",
    "adabelief",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamaxw",
    "adamw",
    "amsgrad",
    "fromage",
    "lamb",
    "lars",
    "lbfgs",
    "lion",
    "noisy_sgd",
    "novograd",
    "optimistic_gradient_descent",
    "polyak_sgd",
    "radam",
    "rmsprop",
    "rprop",
    "scale_by_adamax",
    "scale_by_factored_rms",
    "scale_by_lbfgs",
    "scale_by_optimistic_gradient",
    "scale_by_polyak",
    "scale_by_radam",
    "scale_by_rprop",
    "scale_by_sign",
    "scale_by_sm3",
    "sgd",
    "sign_sgd",
    "sm3",
    "yogi",
]
