"""Inject schedules."""

from typing import Any

from typing import Union, Optional, Iterable
import collections.abc
import numpy as np


# A mock for GradientTransformation to match the signature
class GradientTransformation:
    """GradientTransformation."""

    pass


def inject_hyperparams(
    inner_factory: "collections.abc.Callable[..., Any]",  # type: ignore
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[np.dtype] = None,
):
    """Wrapper to injects stateful hyperparameters into GradientTransformations."""
    return inner_factory


def inject_stateful_hyperparams(
    inner_factory: "collections.abc.Callable[..., Any]",  # type: ignore
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[np.dtype] = None,
):
    """Wrapper to injects stateful hyperparameters into GradientTransformations."""
    return inner_factory
