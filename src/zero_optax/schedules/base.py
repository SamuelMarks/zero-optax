"Base schedules."

from typing import Optional, Union, Callable, Dict, Sequence
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]
Numeric = Union[float, int]
Schedule = Callable[[Numeric], Numeric]


def constant_schedule(value: Numeric) -> Schedule:
    """Constructs a constant schedule."""
    pass


def linear_schedule(
    init_value: Numeric,
    end_value: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Schedule with linear transition from ``init_value`` to ``end_value``."""
    pass


def polynomial_schedule(
    init_value: Numeric,
    end_value: Numeric,
    power: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Constructs a schedule with polynomial transition from init to end value."""
    pass


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Constructs a schedule with either continuous or discrete exponential decay."""
    pass


def cosine_decay_schedule(
    init_value: float, decay_steps: int, alpha: float = 0.0, exponent: float = 1.0
) -> Schedule:
    """Returns a function which implements cosine learning rate decay."""
    pass


def piecewise_constant_schedule(
    init_value: float, boundaries_and_scales: Optional[Dict[int, float]] = None
) -> Schedule:
    """Returns a function which implements a piecewise constant schedule."""
    pass


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Optional[Dict[int, float]] = None,
) -> Schedule:
    """Returns a function which implements a piecewise interpolated schedule."""
    pass


def join_schedules(
    schedules: Sequence[Callable[[Numeric], Numeric]], boundaries: Sequence[int]
) -> Schedule:
    """Sequentially apply multiple schedules."""
    pass
