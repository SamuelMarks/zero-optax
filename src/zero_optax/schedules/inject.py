"Inject schedules."

from typing import Any
from typing import Union, Optional, Iterable
import collections.abc
import numpy as np


class GradientTransformation:
    """GradientTransformation."""

    pass


def inject_hyperparams(
    inner_factory: "collections.abc.Callable[..., Any]",
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[np.dtype] = None,
):
    """Wrapper to injects stateful hyperparameters into GradientTransformations."""
    pass


def inject_stateful_hyperparams(
    inner_factory: "collections.abc.Callable[..., Any]",
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[np.dtype] = None,
):
    """Wrapper to injects stateful hyperparameters into GradientTransformations."""
    pass
