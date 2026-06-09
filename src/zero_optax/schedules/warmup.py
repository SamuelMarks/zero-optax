"""Warmup schedules."""

from typing import Optional
from zero_optax.schedules.base import Numeric, Schedule


def warmup_constant_schedule(
    init_value: float, peak_value: float, warmup_steps: int
) -> Schedule:
    """Linear warmup followed by constant schedule i.e no decay."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= warmup_steps and warmup_steps > 0:
            return float(
                init_value + (peak_value - init_value) * (count / warmup_steps)
            )
        return float(peak_value)

    return schedule


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    """Linear warmup followed by cosine decay."""
    import numpy as np

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= warmup_steps and warmup_steps > 0:
            return float(
                init_value + (peak_value - init_value) * (count / warmup_steps)
            )

        count_decay = count - warmup_steps
        fraction = min(count_decay / max(1, decay_steps), 1.0)
        cosine_decayed = 0.5 * (1.0 + np.cos(np.pi * fraction))
        decayed = cosine_decayed**exponent
        return float(end_value + (peak_value - end_value) * decayed)

    return schedule


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
    import numpy as np

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= warmup_steps and warmup_steps > 0:
            return float(
                init_value + (peak_value - init_value) * (count / warmup_steps)
            )

        count_decay = count - warmup_steps
        if count_decay <= transition_begin:
            return float(peak_value)

        p = (count_decay - transition_begin) / transition_steps
        if staircase:
            p = float(np.floor(p))

        val = peak_value * (decay_rate**p)
        if end_value is not None:
            if decay_rate < 1.0:
                val = max(val, end_value)
            else:
                val = min(val, end_value)
        return float(val)

    return schedule
