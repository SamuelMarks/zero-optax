"""Learning rate schedule functions and hyperparameter injection wrappers."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple, Protocol, Union, runtime_checkable

from zero_optax._tree import asarray
from zero_optax._types import (
    DTypeLike,
    MaskOrFn,
    Params,
    ScalarOrSchedule,
    Schedule,
    Tensor,
    Updates,
)
from zero_optax.base import GradientTransformation, OptState


@runtime_checkable
class StatefulScheduleProtocol(Protocol):
    """Protocol for a stateful schedule with `.init()`, `.update()`, and `__call__()`."""

    def init(self) -> OptState:
        """Initialize the internal schedule state.

        Returns:
            Initialized schedule state.
        """
        ...

    def update(
        self,
        state: OptState,
        **extra_args: float | int | str | bool | Tensor | None,
    ) -> OptState:
        """Advance the internal schedule state.

        Args:
            state: Current schedule state.
            **extra_args: Optional extra keyword arguments.

        Returns:
            Updated schedule state.
        """
        ...

    def __call__(
        self,
        state: OptState,
        **extra_args: float | int | str | bool | Tensor | None,
    ) -> float | int | Tensor:
        """Evaluate the schedule at `state`.

        Args:
            state: Current schedule state.
            **extra_args: Optional extra keyword arguments.

        Returns:
            Evaluated hyperparameter value.
        """
        ...


HyperparamArg = Union[
    ScalarOrSchedule,
    StatefulScheduleProtocol,
    DTypeLike,
    MaskOrFn,
    bool,
    None,
]


class OptimizerFactory(Protocol):
    """Protocol for a wrapped optimizer factory accepting schedules or scalars."""

    def __call__(
        self,
        learning_rate: HyperparamArg,
        /,
        *args: HyperparamArg,
        **kwargs: HyperparamArg,
    ) -> GradientTransformation:
        """Construct a `GradientTransformation` from hyperparameters.

        Args:
            learning_rate: Primary learning rate or schedule hyperparameter.
            *args: Positional hyperparameters.
            **kwargs: Keyword hyperparameters.

        Returns:
            Constructed `GradientTransformation`.
        """
        ...


class AdamwFn(Protocol):
    """Structural protocol matching `adamw` signature for `inject_hyperparams`."""

    def __call__(
        self,
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: DTypeLike = None,
        weight_decay: ScalarOrSchedule = 1e-4,
        mask: MaskOrFn = None,
        *,
        nesterov: bool = False,
    ) -> GradientTransformation:
        """Construct an AdamW `GradientTransformation`.

        Args:
            learning_rate: Learning rate or schedule.
            b1: First moment decay.
            b2: Second moment decay.
            eps: Denominator epsilon.
            eps_root: Square-root epsilon.
            mu_dtype: First moment dtype.
            weight_decay: Weight decay strength or schedule.
            mask: Weight decay mask.
            nesterov: Whether to use Nesterov momentum.

        Returns:
            Constructed `GradientTransformation`.
        """
        ...


SgdFn = Callable[
    [ScalarOrSchedule, Union[float, None], bool, DTypeLike],
    GradientTransformation,
]
InnerFactoryType = Union[OptimizerFactory, SgdFn, AdamwFn]


class InjectHyperparamsState(NamedTuple):
    """State for `inject_hyperparams` transformations.

    Attributes:
        count: Step counter Tensor.
        hyperparams: Dictionary of resolved hyperparameter values at the current step.
        inner_state: Optimizer state of the wrapped inner transformation.
    """

    count: Tensor
    hyperparams: dict[str, float | int | Tensor]
    inner_state: OptState


class InjectStatefulHyperparamsState(NamedTuple):
    """State for `inject_stateful_hyperparams` transformations.

    Attributes:
        count: Step counter Tensor.
        hyperparams: Dictionary of resolved hyperparameter values at the current step.
        hyperparams_states: Dictionary of internal states for stateful schedules.
        inner_state: Optimizer state of the wrapped inner transformation.
    """

    count: Tensor
    hyperparams: dict[str, float | int | Tensor]
    hyperparams_states: dict[str, OptState]
    inner_state: OptState


def _step_to_float(step: int | float | Tensor) -> float:
    """Convert an integer, float, or scalar Tensor step to a Python float.

    Args:
        step: Step count value.

    Returns:
        Floating-point step representation.
    """
    if isinstance(step, Tensor):
        return float(step.item())
    return float(step)


def constant_schedule(value: float | int) -> Schedule:
    """Create a schedule that returns a constant `value` for every step.

    Args:
        value: Constant scalar learning rate returned at all steps.

    Returns:
        A callable schedule mapping `step` to `value`.
    """
    const_val = float(value)

    def schedule(step: int | Tensor) -> float:
        """Evaluate the constant schedule at `step`.

        Args:
            step: Current training step count (unused).

        Returns:
            Constant value `value`.
        """
        del step
        return const_val

    return schedule


def join_schedules(
    schedules: Sequence[Schedule],
    boundaries: Sequence[int],
) -> Schedule:
    """Sequentially compose multiple schedules across global step `boundaries`.

    Each schedule receives a step count indicating the number of steps since the
    previous boundary transition.

    Args:
        schedules: Sequence of callable schedule functions.
        boundaries: Sequence of cumulative step boundaries of length `len(schedules) - 1`.

    Returns:
        A composite schedule callable switching sub-schedules at `boundaries`.
    """

    def schedule(step: int | Tensor) -> float | Tensor:
        """Evaluate the active sub-schedule for `step`.

        Args:
            step: Global training step count.

        Returns:
            Scheduled value from the active segment.
        """
        step_f = _step_to_float(step)
        output: float | Tensor = schedules[0](step)
        for boundary, sub_schedule in zip(boundaries, schedules[1:]):
            if step_f >= float(boundary):
                rel_step = (
                    step - boundary
                    if isinstance(step, Tensor)
                    else int(step_f - float(boundary))
                )
                output = sub_schedule(rel_step)
        return output

    return schedule


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Create a schedule that linearly interpolates from `init_value` to `end_value`.

    Args:
        init_value: Initial value before and at `transition_begin`.
        end_value: Target value reached after `transition_steps`.
        transition_steps: Number of steps over which the linear transition takes place.
        transition_begin: Step index at which the transition starts.

    Returns:
        A callable schedule implementing the linear interpolation.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the linear schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Linearly interpolated scalar value.
        """
        step_f = _step_to_float(step)
        if step_f < float(transition_begin) or transition_steps <= 0:
            return float(init_value)
        step_val = min(step_f - float(transition_begin), float(transition_steps))
        frac = step_val / float(transition_steps)
        return float(init_value + (end_value - init_value) * frac)

    return schedule


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Create a schedule that polynomially decays from `init_value` to `end_value`.

    Args:
        init_value: Initial value before and at `transition_begin`.
        end_value: Final value after `transition_steps`.
        power: Polynomial exponent (`1.0` corresponds to linear interpolation).
        transition_steps: Number of steps over which the polynomial decay takes place.
        transition_begin: Step index at which the transition starts.

    Returns:
        A callable schedule implementing the polynomial transition.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the polynomial schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Polynomially interpolated scalar value.
        """
        step_f = _step_to_float(step)
        if step_f < float(transition_begin) or transition_steps <= 0:
            return float(init_value)
        step_val = min(step_f - float(transition_begin), float(transition_steps))
        p = step_val / float(transition_steps)
        return float((init_value - end_value) * ((1.0 - p) ** power) + end_value)

    return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: float | None = None,
) -> Schedule:
    """Create an exponential decay learning rate schedule.

    Args:
        init_value: Initial value before decay begins.
        transition_steps: Number of steps per `decay_rate` factor application.
        decay_rate: Multiplicative base of the exponential decay.
        transition_begin: Step index at which decay begins.
        staircase: If True, apply discrete step drops using `floor`.
        end_value: Optional lower (when `decay_rate < 1`) or upper (when `decay_rate >= 1`) bound.

    Returns:
        A callable schedule implementing exponential decay.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the exponential decay schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Exponentially decayed scalar learning rate.
        """
        if transition_steps <= 0 or decay_rate <= 0.0:
            return float(init_value)
        step_f = _step_to_float(step)
        if step_f < float(transition_begin):
            return float(init_value)
        p = (step_f - float(transition_begin)) / float(transition_steps)
        if staircase:
            p = float(math.floor(p))
        val = float(init_value) * math.pow(decay_rate, p)
        if end_value is not None:
            if decay_rate < 1.0:
                return max(val, float(end_value))
            return min(val, float(end_value))
        return val

    return schedule


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    """Create a cosine annealing decay schedule.

    Args:
        init_value: Initial learning rate at step 0.
        decay_steps: Total number of steps over which cosine decay occurs.
        alpha: Minimum value multiplier as a fraction of `init_value`.
        exponent: Exponent applied to the `[0, 1]` cosine decay factor.

    Returns:
        A callable schedule implementing cosine decay.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the cosine decay schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Cosine-decayed scalar learning rate.
        """
        eff_decay = max(1, decay_steps)
        step_f = min(max(0.0, _step_to_float(step)), float(eff_decay))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step_f / float(eff_decay)))
        cosine_decay = cosine_decay**exponent
        decayed = (1.0 - alpha) * cosine_decay + alpha
        return float(init_value * decayed)

    return schedule


def warmup_constant_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
) -> Schedule:
    """Create a schedule with linear warmup from `init_value` to a constant `peak_value`.

    Args:
        init_value: Starting learning rate at step 0.
        peak_value: Constant learning rate maintained after `warmup_steps`.
        warmup_steps: Number of steps for the linear warmup phase.

    Returns:
        A callable schedule implementing warmup followed by a constant plateau.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the warmup constant schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Scheduled learning rate at `step`.
        """
        step_f = _step_to_float(step)
        if warmup_steps > 0 and step_f < float(warmup_steps):
            return float(
                init_value + (peak_value - init_value) * (step_f / float(warmup_steps))
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
    """Create a linear warmup followed by cosine decay learning rate schedule.

    Note that matching `optax.warmup_cosine_decay_schedule`, `decay_steps` is the
    total length of the schedule including `warmup_steps`, so the cosine decay
    phase runs for `decay_steps - warmup_steps` steps.

    Args:
        init_value: Starting value at step 0.
        peak_value: Maximum value reached at `warmup_steps`.
        warmup_steps: Number of steps for the linear warmup phase.
        decay_steps: Total number of steps (including `warmup_steps`) over which the schedule runs.
        end_value: Final learning rate at the end of the cosine decay phase.
        exponent: Exponent applied to the cosine decay curve.

    Returns:
        A callable schedule implementing warmup + cosine decay.
    """
    alpha = 0.0 if peak_value == 0.0 else (end_value / peak_value)
    cosine_steps = max(1, decay_steps - warmup_steps)
    schedules = [
        linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=cosine_steps,
            alpha=alpha,
            exponent=exponent,
        ),
    ]
    return join_schedules(schedules, [warmup_steps])


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: float | None = None,
) -> Schedule:
    """Create a linear warmup followed by exponential decay learning rate schedule.

    Args:
        init_value: Initial value at step 0.
        peak_value: Target value at the end of warmup.
        warmup_steps: Number of steps for linear warmup.
        transition_steps: Number of steps per `decay_rate` application.
        decay_rate: Base of the exponential decay.
        transition_begin: Additional delay steps after warmup before decay starts.
        staircase: Whether to apply discrete staircase drops.
        end_value: Optional lower or upper bound on the decayed value.

    Returns:
        A callable schedule implementing warmup + exponential decay.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the warmup exponential decay schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Scheduled learning rate at `step`.
        """
        step_f = _step_to_float(step)
        if warmup_steps > 0 and step_f < float(warmup_steps):
            return float(
                init_value + (peak_value - init_value) * (step_f / float(warmup_steps))
            )
        p = (step_f - float(warmup_steps) - float(transition_begin)) / float(
            transition_steps
        )
        if p < 0.0:
            return float(peak_value)
        if staircase:
            p = float(math.floor(p))
        val = float(peak_value) * math.pow(decay_rate, p)
        if end_value is not None:
            if decay_rate < 1.0:
                return max(val, float(end_value))
            return min(val, float(end_value))
        return val

    return schedule


def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: Mapping[int, float] | None = None,
) -> Schedule:
    """Create a piecewise constant schedule multiplying by scale factors at boundaries.

    Args:
        init_value: Initial value before the first boundary.
        boundaries_and_scales: Optional mapping `{step_boundary: scale_multiplier}`.

    Returns:
        A callable schedule implementing piecewise constant step drops.
    """

    def schedule(step: int | Tensor) -> float:
        """Evaluate the piecewise constant schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Scaled learning rate after applying all crossed boundary multipliers.
        """
        val = float(init_value)
        if boundaries_and_scales is None:
            return val
        step_f = _step_to_float(step)
        for boundary, scale_factor in sorted(boundaries_and_scales.items()):
            if step_f >= float(boundary):
                val *= float(scale_factor)
        return val

    return schedule


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Mapping[int, float] | None = None,
) -> Schedule:
    """Create a schedule that interpolates between cumulative scale products at boundaries.

    Args:
        interpolate_type: Interpolation mode: `'linear'`, `'cosine'`, or `'none'`.
        init_value: Initial value at step 0.
        boundaries_and_scales: Optional mapping `{step_boundary: scale_factor}` where
            each `scale_factor` multiplies the previous boundary value.

    Returns:
        A callable schedule implementing piecewise interpolation.

    Raises:
        ValueError: If `interpolate_type` is not `'linear'`, `'cosine'`, or `'none'`,
            or if any scale factor is negative.
    """
    if interpolate_type not in ("linear", "cosine", "none"):
        raise ValueError(
            "`interpolate_type` must be 'linear', 'cosine', or 'none', "
            f"got {interpolate_type!r}."
        )
    if boundaries_and_scales:
        sorted_items = sorted(boundaries_and_scales.items(), key=lambda kv: kv[0])
        if not all(float(scale) >= 0.0 for _, scale in sorted_items):
            raise ValueError(
                "`piecewise_interpolate_schedule` expects non-negative scale factors."
            )
    else:
        sorted_items = []

    boundaries: list[int] = [0]
    values: list[float] = [float(init_value)]
    cur_val = float(init_value)
    for b_step, scale_factor in sorted_items:
        boundaries.append(int(b_step))
        cur_val = cur_val * float(scale_factor)
        values.append(cur_val)

    def schedule(step: int | Tensor) -> float:
        """Evaluate the piecewise interpolated schedule at `step`.

        Args:
            step: Current training step count.

        Returns:
            Interpolated learning rate at `step`.
        """
        if not sorted_items:
            return float(init_value)

        step_f = _step_to_float(step)
        if step_f < 0.0:
            return values[0]

        for idx in range(len(boundaries) - 1):
            b_start = float(boundaries[idx])
            b_end = float(boundaries[idx + 1])
            if b_start <= step_f < b_end:
                if interpolate_type == "none" or b_end <= b_start:
                    return values[idx]
                pct = (step_f - b_start) / (b_end - b_start)
                if interpolate_type == "cosine":
                    pct = 0.5 * (1.0 - math.cos(math.pi * pct))
                return float(values[idx] + pct * (values[idx + 1] - values[idx]))

        return values[-1]

    return schedule


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Create a 2-phase cosine 1-cycle learning rate schedule.

    Args:
        transition_steps: Total number of steps in the cycle.
        peak_value: Maximum learning rate reached at `pct_start * transition_steps`.
        pct_start: Fraction of total cycle steps spent warming up.
        div_factor: Ratio determining `init_value = peak_value / div_factor`.
        final_div_factor: Ratio determining `end_value = init_value / final_div_factor`.

    Returns:
        A callable schedule implementing the cosine 1-cycle policy.

    Raises:
        ValueError: If `transition_steps <= 0`.
    """
    if transition_steps <= 0:
        raise ValueError(
            "A cosine onecycle schedule was set with a non-positive `transition_steps`."
        )
    return piecewise_interpolate_schedule(
        "cosine",
        peak_value / div_factor,
        {
            int(pct_start * transition_steps): div_factor,
            int(transition_steps): 1.0 / (div_factor * final_div_factor),
        },
    )


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Create a 3-phase piecewise linear 1-cycle learning rate schedule.

    Args:
        transition_steps: Total number of steps in the cycle.
        peak_value: Maximum learning rate at the end of warmup.
        pct_start: Fraction of `transition_steps` at which warmup ends.
        pct_final: Fraction of `transition_steps` at which the main linear decay ends.
        div_factor: Factor determining `init_value = peak_value / div_factor`.
        final_div_factor: Factor determining `end_value = init_value / final_div_factor`.

    Returns:
        A callable schedule implementing the 3-phase linear 1-cycle policy.

    Raises:
        ValueError: If `transition_steps <= 0`.
    """
    if transition_steps <= 0:
        raise ValueError(
            "A linear onecycle schedule was set with a non-positive `transition_steps`."
        )
    return piecewise_interpolate_schedule(
        "linear",
        peak_value / div_factor,
        {
            int(pct_start * transition_steps): div_factor,
            int(pct_final * transition_steps): 1.0 / div_factor,
            int(transition_steps): 1.0 / final_div_factor,
        },
    )


def sgdr_schedule(
    cosine_kwargs: Iterable[Mapping[str, float | int]] | None,
) -> Schedule:
    """Create a Stochastic Gradient Descent with Warm Restarts (SGDR) schedule.

    Chains multiple cosine decay (or warmup cosine decay) cycles specified by
    each dictionary in `cosine_kwargs`.

    Args:
        cosine_kwargs: Iterable of dictionaries with cycle parameters (`init_value`,
            `peak_value`, `warmup_steps`, `decay_steps`, `end_value`, `exponent`).

    Returns:
        A callable schedule implementing periodic cosine warm restarts.
    """
    kw_list = list(cosine_kwargs) if cosine_kwargs is not None else []
    if not kw_list:
        return constant_schedule(0.0)

    boundaries: list[int] = []
    schedules: list[Schedule] = []
    step_acc = 0
    for cycle_cfg in kw_list:
        decay_steps = int(cycle_cfg.get("decay_steps", 1))
        init_val = float(cycle_cfg.get("init_value", 0.0))
        peak_val = float(cycle_cfg.get("peak_value", init_val))
        warmup_steps = int(cycle_cfg.get("warmup_steps", 0))
        end_val = float(cycle_cfg.get("end_value", 0.0))
        exponent = float(cycle_cfg.get("exponent", 1.0))
        schedules.append(
            warmup_cosine_decay_schedule(
                init_value=init_val,
                peak_value=peak_val,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_val,
                exponent=exponent,
            )
        )
        boundaries.append(step_acc + decay_steps)
        step_acc += decay_steps

    return join_schedules(schedules, boundaries[:-1])


def _call_factory_with_bound_args(
    inner_factory: InnerFactoryType,
    sig: inspect.Signature,
    call_args: Mapping[str, HyperparamArg],
) -> GradientTransformation:
    """Invoke `inner_factory` reconstructing positional and keyword arguments from `sig`.

    Args:
        inner_factory: Optimizer factory callable.
        sig: Inspected signature of `inner_factory`.
        call_args: Mapping of parameter names to resolved values.

    Returns:
        Constructed `GradientTransformation`.
    """
    pos_args: list[HyperparamArg] = []
    kw_args: dict[str, HyperparamArg] = {}
    for param_name, param in sig.parameters.items():
        val = call_args[param_name]
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_args[param_name] = val
        else:
            pos_args.append(val)
    factory_fn: OptimizerFactory = inner_factory
    return factory_fn(pos_args[0], *pos_args[1:], **kw_args)


def _format_hparam_value(
    raw: float | int | Tensor,
    hyperparam_dtype: DTypeLike,
) -> float | int | Tensor:
    """Convert a resolved hyperparameter value according to `hyperparam_dtype`.

    Args:
        raw: Evaluated scalar or Tensor hyperparameter value.
        hyperparam_dtype: Optional target dtype for casting.

    Returns:
        Formatted hyperparameter value.
    """
    if hyperparam_dtype is not None:
        return asarray(raw, dtype=hyperparam_dtype)
    return raw


def _resolve_hyperparams_and_states(
    bound_args: Mapping[str, HyperparamArg],
    static_set: set[str],
    count: Tensor,
    hyperparam_dtype: DTypeLike,
    prev_states: Mapping[str, OptState] | None = None,
) -> tuple[dict[str, float | int | Tensor], dict[str, OptState]]:
    """Evaluate all non-static schedule and stateful hyperparameters at step `count`.

    Args:
        bound_args: Bound argument mapping from the wrapped optimizer factory.
        static_set: Argument names that should remain static.
        count: Current step counter Tensor.
        hyperparam_dtype: Optional dtype to cast resolved tensor hyperparameters to.
        prev_states: Previous stateful schedule states (None during `init`).

    Returns:
        Tuple `(resolved_hyperparams, next_hyperparam_states)`.
    """
    resolved: dict[str, float | int | Tensor] = {}
    next_states: dict[str, OptState] = {}
    for name, val in bound_args.items():
        if name in static_set or isinstance(val, bool):
            continue
        if isinstance(val, StatefulScheduleProtocol):
            cur_st = (
                val.init()
                if prev_states is None or name not in prev_states
                else prev_states[name]
            )
            raw_val = val(cur_st)
            next_st = val.update(cur_st) if prev_states is not None else cur_st
            next_states[name] = next_st
            resolved[name] = _format_hparam_value(raw_val, hyperparam_dtype)
        elif callable(val):
            raw_sched = val(count)
            if isinstance(raw_sched, (int, float, Tensor)):
                resolved[name] = _format_hparam_value(raw_sched, hyperparam_dtype)
        elif isinstance(val, (int, float, Tensor)):
            resolved[name] = _format_hparam_value(val, hyperparam_dtype)
    return resolved, next_states


def inject_hyperparams(
    inner_factory: InnerFactoryType | None,
    static_args: str | Iterable[str] = (),
    hyperparam_dtype: DTypeLike = None,
) -> OptimizerFactory | None:
    """Wrap an optimizer factory so its hyperparameters are tracked in `InjectHyperparamsState`.

    Args:
        inner_factory: Optimizer factory callable returning a `GradientTransformation`,
            or None.
        static_args: Parameter name or iterable of parameter names to treat as static.
        hyperparam_dtype: Optional data type for storing resolved hyperparameters in state.

    Returns:
        Wrapped factory callable accepting schedules in place of scalar hyperparameters,
        or None if `inner_factory` is None.

    Raises:
        ValueError: If `static_args` contains names not in `inner_factory`'s signature.
    """
    if inner_factory is None:
        return None
    static_set: set[str] = (
        {static_args} if isinstance(static_args, str) else set(static_args)
    )
    sig = inspect.signature(inner_factory)
    if not static_set.issubset(sig.parameters.keys()):
        raise ValueError(
            "`static_args` must specify a subset of `inner_factory`'s parameters. "
            f"Given `static_args`: {static_set}. `inner_factory` parameters: "
            f"{set(sig.parameters.keys())}"
        )

    def wrapped_factory(
        learning_rate: HyperparamArg,
        /,
        *args: HyperparamArg,
        **kwargs: HyperparamArg,
    ) -> GradientTransformation:
        """Instantiate the wrapped transformation with dynamic hyperparameter tracking.

        Args:
            learning_rate: Primary learning rate or hyperparameter.
            *args: Positional arguments forwarded to `inner_factory`.
            **kwargs: Keyword arguments forwarded to `inner_factory`.

        Returns:
            A `GradientTransformation` tracking `InjectHyperparamsState`.
        """
        bound = sig.bind_partial(learning_rate, *args, **kwargs)
        bound.apply_defaults()
        bound_dict: dict[str, HyperparamArg] = dict(bound.arguments)

        def init_fn(params: Params) -> InjectHyperparamsState:
            """Initialize `InjectHyperparamsState` at step 0.

            Args:
                params: Model parameters PyTree.

            Returns:
                Initialized `InjectHyperparamsState`.
            """
            count0 = asarray(0)
            hparams, _ = _resolve_hyperparams_and_states(
                bound_dict, static_set, count0, hyperparam_dtype
            )
            call_args: dict[str, HyperparamArg] = dict(bound_dict)
            for k, v in hparams.items():
                call_args[k] = v
            init_tx = _call_factory_with_bound_args(inner_factory, sig, call_args)
            return InjectHyperparamsState(
                count=count0,
                hyperparams=hparams,
                inner_state=init_tx.init(params),
            )

        def update_fn(
            updates: Updates,
            state: OptState,
            params: Params | None = None,
        ) -> tuple[Updates, InjectHyperparamsState]:
            """Evaluate scheduled hyperparameters at `state.count` and run inner update.

            Args:
                updates: Gradient updates PyTree.
                state: Current `InjectHyperparamsState`.
                params: Optional model parameters PyTree.

            Returns:
                Tuple of `(transformed_updates, new_inject_state)`.
            """
            if not isinstance(state, InjectHyperparamsState):
                state = init_fn(updates)
            count = state.count
            hparams: dict[str, float | int | Tensor] = {
                k: _format_hparam_value(v, hyperparam_dtype)
                for k, v in state.hyperparams.items()
            }
            sched_hparams, _ = _resolve_hyperparams_and_states(
                bound_dict, static_set, count, hyperparam_dtype
            )
            for k, orig_val in bound_dict.items():
                if k not in static_set and callable(orig_val) and k in sched_hparams:
                    hparams[k] = sched_hparams[k]

            call_args: dict[str, HyperparamArg] = dict(bound_dict)
            for k, v in hparams.items():
                call_args[k] = v
            step_tx = _call_factory_with_bound_args(inner_factory, sig, call_args)
            new_updates, new_inner_state = step_tx.update(
                updates, state.inner_state, params
            )
            return new_updates, InjectHyperparamsState(
                count=count + 1,
                hyperparams=hparams,
                inner_state=new_inner_state,
            )

        return GradientTransformation(init_fn, update_fn)

    return wrapped_factory


def inject_stateful_hyperparams(
    inner_factory: InnerFactoryType | None,
    static_args: str | Iterable[str] = (),
    hyperparam_dtype: DTypeLike = None,
) -> OptimizerFactory | None:
    """Wrap an optimizer factory to inject stateful schedules into `InjectStatefulHyperparamsState`.

    Args:
        inner_factory: Optimizer factory callable returning a `GradientTransformation`,
            or None.
        static_args: Parameter name or iterable of parameter names to treat as static.
        hyperparam_dtype: Optional data type for storing resolved hyperparameters in state.

    Returns:
        Wrapped factory callable maintaining `InjectStatefulHyperparamsState`,
        or None if `inner_factory` is None.

    Raises:
        ValueError: If `static_args` contains names not in `inner_factory`'s signature.
    """
    if inner_factory is None:
        return None
    static_set: set[str] = (
        {static_args} if isinstance(static_args, str) else set(static_args)
    )
    sig = inspect.signature(inner_factory)
    if not static_set.issubset(sig.parameters.keys()):
        raise ValueError(
            "`static_args` must specify a subset of `inner_factory`'s parameters. "
            f"Given `static_args`: {static_set}. `inner_factory` parameters: "
            f"{set(sig.parameters.keys())}"
        )

    def wrapped_factory(
        learning_rate: HyperparamArg,
        /,
        *args: HyperparamArg,
        **kwargs: HyperparamArg,
    ) -> GradientTransformation:
        """Instantiate the stateful hyperparameter-injected transformation.

        Args:
            learning_rate: Primary learning rate or hyperparameter.
            *args: Positional arguments forwarded to `inner_factory`.
            **kwargs: Keyword arguments forwarded to `inner_factory`.

        Returns:
            A `GradientTransformation` maintaining `InjectStatefulHyperparamsState`.
        """
        bound = sig.bind_partial(learning_rate, *args, **kwargs)
        bound.apply_defaults()
        bound_dict: dict[str, HyperparamArg] = dict(bound.arguments)

        def init_fn(params: Params) -> InjectStatefulHyperparamsState:
            """Initialize `InjectStatefulHyperparamsState` at step 0.

            Args:
                params: Model parameters PyTree.

            Returns:
                Initialized `InjectStatefulHyperparamsState`.
            """
            count0 = asarray(0)
            hparams, hparam_states = _resolve_hyperparams_and_states(
                bound_dict, static_set, count0, hyperparam_dtype, prev_states=None
            )
            call_args: dict[str, HyperparamArg] = dict(bound_dict)
            for k, v in hparams.items():
                call_args[k] = v
            init_tx = _call_factory_with_bound_args(inner_factory, sig, call_args)
            return InjectStatefulHyperparamsState(
                count=count0,
                hyperparams=hparams,
                hyperparams_states=hparam_states,
                inner_state=init_tx.init(params),
            )

        def update_fn(
            updates: Updates,
            state: OptState,
            params: Params | None = None,
        ) -> tuple[Updates, InjectStatefulHyperparamsState]:
            """Evaluate stateful schedules at `state.count` and run inner update.

            Args:
                updates: Gradient updates PyTree.
                state: Current `InjectStatefulHyperparamsState`.
                params: Optional model parameters PyTree.

            Returns:
                Tuple of `(transformed_updates, new_stateful_inject_state)`.
            """
            if not isinstance(state, InjectStatefulHyperparamsState):
                state = init_fn(updates)
            count = state.count
            hparams: dict[str, float | int | Tensor] = {
                k: _format_hparam_value(v, hyperparam_dtype)
                for k, v in state.hyperparams.items()
            }
            sched_hparams, next_hparam_states = _resolve_hyperparams_and_states(
                bound_dict,
                static_set,
                count,
                hyperparam_dtype,
                prev_states=state.hyperparams_states,
            )
            for k, orig_val in bound_dict.items():
                if (
                    k not in static_set
                    and (
                        callable(orig_val)
                        or isinstance(orig_val, StatefulScheduleProtocol)
                    )
                    and k in sched_hparams
                ):
                    hparams[k] = sched_hparams[k]

            call_args: dict[str, HyperparamArg] = dict(bound_dict)
            for k, v in hparams.items():
                call_args[k] = v
            step_tx = _call_factory_with_bound_args(inner_factory, sig, call_args)
            new_updates, new_inner_state = step_tx.update(
                updates, state.inner_state, params
            )
            return new_updates, InjectStatefulHyperparamsState(
                count=count + 1,
                hyperparams=hparams,
                hyperparams_states=next_hparam_states,
                inner_state=new_inner_state,
            )

        return GradientTransformation(init_fn, update_fn)

    return wrapped_factory


__all__ = [
    "InjectHyperparamsState",
    "InjectStatefulHyperparamsState",
    "StatefulScheduleProtocol",
    "constant_schedule",
    "cosine_decay_schedule",
    "cosine_onecycle_schedule",
    "exponential_decay",
    "inject_hyperparams",
    "inject_stateful_hyperparams",
    "join_schedules",
    "linear_onecycle_schedule",
    "linear_schedule",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polynomial_schedule",
    "sgdr_schedule",
    "warmup_constant_schedule",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
]
