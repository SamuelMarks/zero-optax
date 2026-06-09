"""Cycle schedules."""

from typing import Iterable, Dict, Any
import numpy as np
from zero_optax.schedules.base import Numeric, Schedule


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Schedule:
    """Returns a function which implements the onecycle learning rate schedule."""
    init_val = peak_value / div_factor
    final_val = init_val / final_div_factor
    warmup_steps = int(transition_steps * pct_start)
    decay_steps = transition_steps - warmup_steps

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= warmup_steps:
            # Ascend using cosine
            fraction = count / max(1, warmup_steps)
            # phase shifts to start from init_val to peak_value
            cosine_val = 0.5 * (1.0 - np.cos(np.pi * fraction))
            return float(init_val + (peak_value - init_val) * cosine_val)
        else:
            # Descend
            fraction = min((count - warmup_steps) / max(1, decay_steps), 1.0)
            cosine_val = 0.5 * (1.0 + np.cos(np.pi * fraction))
            return float(final_val + (peak_value - final_val) * cosine_val)

    return schedule


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Schedule:
    """Returns a learning rate with three linear phases."""
    init_val = peak_value / div_factor
    final_val = init_val / final_div_factor

    phase1_steps = int(transition_steps * pct_start)
    phase2_steps = int(transition_steps * pct_final) - phase1_steps
    phase3_steps = transition_steps - phase1_steps - phase2_steps

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = float(count)
        if count <= phase1_steps:
            fraction = count / max(1, phase1_steps)
            return float(init_val + fraction * (peak_value - init_val))
        elif count <= phase1_steps + phase2_steps:
            fraction = (count - phase1_steps) / max(1, phase2_steps)
            return float(peak_value + fraction * (init_val - peak_value))
        else:
            count_phase3 = min(count - phase1_steps - phase2_steps, phase3_steps)
            fraction = count_phase3 / max(1, phase3_steps)
            return float(init_val + fraction * (final_val - init_val))

    return schedule


def sgdr_schedule(cosine_kwargs: Iterable[Dict[str, Any]]) -> Schedule:
    """SGD with warm restarts."""
    # Build a list of schedules and boundaries
    # Each kwarg dict represents a cosine_decay_schedule phase
    from zero_optax.schedules.base import cosine_decay_schedule

    schedules = []
    boundaries = []
    current_boundary = 0

    for kwarg in cosine_kwargs:
        schedules.append(cosine_decay_schedule(**kwarg))
        current_boundary += kwarg.get("decay_steps", 1)
        boundaries.append(current_boundary)

    # We drop the last boundary for join_schedules
    if boundaries:
        boundaries.pop()

    def schedule(count: Numeric) -> Numeric:
        """Schedule fn."""
        count = int(count)
        if not schedules:
            return 0.0
        for i, b in enumerate(boundaries):
            if count < b:
                # Need to offset the count for the phase
                offset = 0 if i == 0 else boundaries[i - 1]
                return float(schedules[i](count - offset))
        offset = 0 if not boundaries else boundaries[-1]
        return float(schedules[-1](count - offset))

    return schedule
