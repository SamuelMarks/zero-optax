"""Base schedules."""

from typing import Optional, Union, Callable, Dict, Sequence
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]
Numeric = Union[float, int]
Schedule = Callable[[Numeric], Numeric]


def constant_schedule(value: Numeric) -> Schedule:
    """Constructs a constant schedule."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        return value

    return schedule


def linear_schedule(
    init_value: Numeric,
    end_value: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Schedule with linear transition from ``init_value`` to ``end_value``."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= transition_begin:
            return float(init_value)
        if count >= transition_begin + transition_steps:
            return float(end_value)

        fraction = (count - transition_begin) / transition_steps
        return float(init_value + fraction * (end_value - init_value))

    return schedule


def polynomial_schedule(
    init_value: Numeric,
    end_value: Numeric,
    power: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Constructs a schedule with polynomial transition from init to end value."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= transition_begin:
            return float(init_value)
        if count >= transition_begin + transition_steps:
            return float(end_value)

        fraction = (count - transition_begin) / transition_steps
        return float((init_value - end_value) * ((1.0 - fraction) ** power) + end_value)

    return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Constructs a schedule with either continuous or discrete exponential decay."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= transition_begin:
            return float(init_value)

        p = (count - transition_begin) / transition_steps
        if staircase:
            p = float(np.floor(p))

        val = init_value * (decay_rate**p)
        if end_value is not None:
            # Optax actually clamps the exponential decay to end_value if specified
            # Note: if decay_rate < 1, it's a lower bound. If > 1, upper bound.
            if decay_rate < 1.0:
                val = max(val, end_value)
            else:
                val = min(val, end_value)
        return float(val)

    return schedule


def cosine_decay_schedule(
    init_value: float, decay_steps: int, alpha: float = 0.0, exponent: float = 1.0
) -> Schedule:
    """Returns a function which implements cosine learning rate decay."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        fraction = min(count / max(1, decay_steps), 1.0)
        cosine_decayed = 0.5 * (1.0 + np.cos(np.pi * fraction))
        decayed = (1.0 - alpha) * (cosine_decayed**exponent) + alpha
        return float(init_value * decayed)

    return schedule


def piecewise_constant_schedule(
    init_value: float, boundaries_and_scales: Optional[Dict[int, float]] = None
) -> Schedule:
    """Returns a function which implements a piecewise constant schedule."""
    if boundaries_and_scales is None:
        boundaries_and_scales = {}

    boundaries = sorted(list(boundaries_and_scales.keys()))
    scales = [boundaries_and_scales[b] for b in boundaries]

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = int(count)
        scale = 1.0
        for b, s in zip(boundaries, scales):
            if count >= b:
                scale = s
        return float(init_value * scale)

    return schedule


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Optional[Dict[int, float]] = None,
) -> Schedule:
    """Returns a function which implements a piecewise interpolated schedule."""
    if boundaries_and_scales is None:
        boundaries_and_scales = {}

    boundaries = sorted(list(boundaries_and_scales.keys()))
    scales = [boundaries_and_scales[b] for b in boundaries]

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if not boundaries:
            return float(init_value)

        if count <= boundaries[0]:
            # Interpolate between init_value and first scale
            fraction = count / max(1, boundaries[0])
            start_val = init_value
            end_val = init_value * scales[0]
        else:
            # Find the segment
            idx = 0
            for i, b in enumerate(boundaries):
                if count >= b:
                    idx = i

            if idx == len(boundaries) - 1:
                return float(init_value * scales[-1])

            start_b = boundaries[idx]
            end_b = boundaries[idx + 1]
            fraction = (count - start_b) / max(1, end_b - start_b)
            start_val = init_value * scales[idx]
            end_val = init_value * scales[idx + 1]

        if interpolate_type == "linear":
            return float(start_val + fraction * (end_val - start_val))
        else:
            # fallback to constant if unrecognized
            return float(start_val)

    return schedule


def join_schedules(
    schedules: Sequence[Callable[[Numeric], Numeric]], boundaries: Sequence[int]
) -> Schedule:
    """Sequentially apply multiple schedules."""

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = int(count)
        for i, b in enumerate(boundaries):
            if count < b:
                return schedules[i](count)
        return schedules[-1](count)

    return schedule
