"Warmup schedules."

from typing import Optional
from zero_optax.schedules.base import Schedule


def warmup_constant_schedule(
    init_value: float, peak_value: float, warmup_steps: int
) -> Schedule:
    """Linear warmup followed by constant schedule i.e no decay."""
    pass


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    """Linear warmup followed by cosine decay."""
    pass


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Linear warmup followed by exponential decay."""
    pass
