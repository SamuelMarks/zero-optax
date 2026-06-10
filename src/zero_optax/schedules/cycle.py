"Cycle schedules."

from typing import Iterable, Dict, Any
from zero_optax.schedules.base import Schedule


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Returns a function which implements the onecycle learning rate schedule."""
    pass


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Returns a learning rate with three linear phases."""
    pass


def sgdr_schedule(cosine_kwargs: Iterable[Dict[str, Any]]) -> Schedule:
    """SGD with warm restarts."""
    pass
