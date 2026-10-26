"""Backtracking and zoom line-search gradient transformations and `value_and_grad_from_state`."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, Protocol, Union

from typing_extensions import TypeGuard

from zero_optax._tree import (
    arange,
    asarray,
    reshape,
    tree_leaves,
    tree_zeros_like,
    where,
)
from zero_optax._types import (
    ArrayLike,
    Params,
    PyTree,
    Scalar,
    Tensor,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    ExtraArgValue,
    GradientTransformationExtraArgs,
    OptState,
)
from zero_optax.projections._projections import _is_scalar_bound
from zero_optax.tree_utils._random import _unflatten_like
from zero_optax.tree_utils._state_utils import tree_get
from zero_optax.tree_utils._tree_math import (
    _flatten_tensor_floats,
    tree_add_scale,
    tree_scalar_mul,
    tree_vdot,
)


class ValueFn0(Protocol):
    """Protocol for a 1-arg objective function `fn(params)`."""

    def __call__(self, params: Params, /) -> Union[Scalar, Tensor]:
        """Evaluate objective at `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Scalar objective value.
        """
        ...


class ValueFnKw(Protocol):
    """Protocol for an objective function accepting keyword arguments."""

    def __call__(self, params: Params, /, **kwargs: PyTree) -> Union[Scalar, Tensor]:
        """Evaluate objective at `params` with keyword arguments.

        Args:
            params: Model parameters PyTree.
            **kwargs: Extra keyword arguments.

        Returns:
            Scalar objective value.
        """
        ...


ValueFn = Union[
    ValueFn0,
    ValueFnKw,
    Callable[[Params], Union[Scalar, Tensor]],
    Callable[[Params, float], Union[Scalar, Tensor]],
    Callable[[Params, PyTree], Union[Scalar, Tensor]],
    Callable[[Params, PyTree, PyTree], Union[Scalar, Tensor]],
]


class ValueAndGradFromStateFn(Protocol):
    """Protocol for the callable returned by `value_and_grad_from_state`."""

    def __call__(
        self,
        params: Params,
        *fn_args: PyTree,
        state: OptState,
        **fn_kwargs: PyTree,
    ) -> tuple[Tensor, Updates]:
        """Fetch `(value, grad)` from `state` or compute them from `value_fn`.

        Args:
            params: Model parameters PyTree.
            *fn_args: Positional arguments forwarded to `value_fn`.
            state: Optimizer state containing `'value'` and `'grad'`.
            **fn_kwargs: Keyword arguments forwarded to `value_fn`.

        Returns:
            Tuple `(value_tensor, grad_pytree)`.
        """
        ...


class BacktrackingLinesearchInfo(NamedTuple):
    """Diagnostics for `scale_by_backtracking_linesearch`.

    Attributes:
        num_linesearch_steps: Number of backtracking steps performed.
        decrease_error: Final Armijo sufficient-decrease violation (`0.0` if satisfied).
    """

    num_linesearch_steps: Tensor
    decrease_error: Tensor


class ScaleByBacktrackingLinesearchState(NamedTuple):
    """State for `scale_by_backtracking_linesearch`.

    Attributes:
        learning_rate: Selected step size Tensor.
        value: Objective value Tensor at the accepted iterate.
        grad: Cached gradient PyTree (or None if `store_grad=False`).
        info: `BacktrackingLinesearchInfo` diagnostic tuple.
    """

    learning_rate: Tensor
    value: Tensor
    grad: Updates
    info: BacktrackingLinesearchInfo


class ZoomLinesearchInfo(NamedTuple):
    """Diagnostics for `scale_by_zoom_linesearch`.

    Attributes:
        num_linesearch_steps: Number of zoom line-search iterations performed.
        decrease_error: Final Armijo decrease violation (`0.0` if satisfied).
        curvature_error: Final Wolfe curvature violation (`0.0` if satisfied).
    """

    num_linesearch_steps: Tensor
    decrease_error: Tensor
    curvature_error: Tensor


class ScaleByZoomLinesearchState(NamedTuple):
    """State for `scale_by_zoom_linesearch`.

    Attributes:
        learning_rate: Selected step size Tensor.
        value: Objective value Tensor at the accepted iterate.
        grad: Cached gradient PyTree at the accepted iterate.
        info: `ZoomLinesearchInfo` diagnostic tuple.
    """

    learning_rate: Tensor
    value: Tensor
    grad: Updates
    info: ZoomLinesearchInfo


def _is_value_fn(val: ExtraArgValue | None) -> TypeGuard[ValueFn]:
    """Check whether `val` is a callable `ValueFn`.

    Args:
        val: Candidate `value_fn` argument.

    Returns:
        True if `val` is callable.
    """
    return callable(val)


def _is_scalar_extra(val: ExtraArgValue | None) -> TypeGuard[ArrayLike]:
    """Check whether `val` is a non-None scalar/tensor `ArrayLike` value.

    Args:
        val: Candidate `value` argument from `extra_args`.

    Returns:
        True if `val` is a non-callable, non-container `ArrayLike`.
    """
    return val is not None and not callable(val) and _is_scalar_bound(val)


def _is_vfn_kw(fn: ValueFn, has_kw: bool) -> TypeGuard[ValueFnKw]:
    """Narrow `fn` to `ValueFnKw` when keyword arguments are present.

    Args:
        fn: Objective function.
        has_kw: Whether keyword arguments are non-empty.

    Returns:
        True if `has_kw`.
    """
    del fn
    return has_kw


def _is_vfn_0(
    fn: ValueFn,
    n_args: int,
) -> TypeGuard[Callable[[Params], Union[Scalar, Tensor]]]:
    """Narrow `fn` to 1-arg `(Params) -> Scalar | Tensor` when `n_args == 0`.

    Args:
        fn: Objective function.
        n_args: Number of extra positional arguments.

    Returns:
        True if `n_args == 0`.
    """
    del fn
    return n_args == 0


def _is_vfn_1(
    fn: ValueFn,
    n_args: int,
) -> TypeGuard[Callable[[Params, PyTree], Union[Scalar, Tensor]]]:
    """Narrow `fn` to 2-arg `(Params, PyTree) -> Scalar | Tensor` when `n_args == 1`.

    Args:
        fn: Objective function.
        n_args: Number of extra positional arguments.

    Returns:
        True if `n_args == 1`.
    """
    del fn
    return n_args == 1


def _is_vfn_2(
    fn: ValueFn,
    n_args: int,
) -> TypeGuard[Callable[[Params, PyTree, PyTree], Union[Scalar, Tensor]]]:
    """Narrow `fn` to 3-arg `(Params, PyTree, PyTree) -> Scalar | Tensor` when `n_args == 2`.

    Args:
        fn: Objective function.
        n_args: Number of extra positional arguments.

    Returns:
        True if `n_args == 2`.
    """
    del fn
    return n_args == 2


def _call_value_fn(
    value_fn: ValueFn,
    params: Params,
    fn_args: Sequence[PyTree] = (),
    fn_kwargs: Mapping[str, PyTree] | None = None,
) -> Union[Scalar, Tensor]:
    """Invoke `value_fn` with `params`, `fn_args`, and `fn_kwargs`.

    Args:
        value_fn: Objective function.
        params: Model parameters PyTree.
        fn_args: Positional arguments.
        fn_kwargs: Keyword arguments.

    Returns:
        Scalar objective value.

    Raises:
        ValueError: If more than 2 positional arguments are supplied in `fn_args`.
    """
    kw = dict(fn_kwargs) if fn_kwargs is not None else {}
    if _is_vfn_kw(value_fn, bool(kw)):
        return value_fn(params, **kw)
    if _is_vfn_0(value_fn, len(fn_args)):
        return value_fn(params)
    if _is_vfn_1(value_fn, len(fn_args)):
        return value_fn(params, fn_args[0])
    if _is_vfn_2(value_fn, len(fn_args)):
        return value_fn(params, fn_args[0], fn_args[1])
    raise ValueError("Unsupported value_fn signature.")


def _filter_fn_kwargs(
    fn: ValueFn,
    extra_args: Mapping[str, ExtraArgValue],
) -> dict[str, PyTree]:
    """Filter `extra_args` to only the keyword arguments accepted by `fn`.

    Args:
        fn: Objective function `value_fn`.
        extra_args: Extra keyword arguments passed to `update`.

    Returns:
        Dictionary of keyword arguments accepted by `fn`.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return {
            k: v
            for k, v in extra_args.items()
            if k not in ("value", "grad", "value_fn") and not callable(v)
        }
    accepted = {p.name for p in params[1:]}
    return {k: v for k, v in extra_args.items() if k in accepted and not callable(v)}


def _numerical_value_and_grad(
    value_fn: ValueFn,
    params: Params,
    fn_args: Sequence[PyTree] = (),
    fn_kwargs: Mapping[str, PyTree] | None = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Updates]:
    """Compute `(value_fn(params, ...), grad(value_fn)(params, ...))` via central finite differences.

    Args:
        value_fn: Scalar objective function `value_fn(params, *fn_args, **fn_kwargs)`.
        params: Model parameters PyTree.
        fn_args: Positional arguments forwarded to `value_fn`.
        fn_kwargs: Keyword arguments forwarded to `value_fn`.
        eps: Finite-difference step size.

    Returns:
        Tuple `(val_tensor, grad_pytree)` matching `params`.
    """
    kw = dict(fn_kwargs) if fn_kwargs is not None else {}
    val_raw = _call_value_fn(value_fn, params, fn_args, kw)
    val_t = asarray(val_raw)

    raw_leaves = tree_leaves(params)
    flat_vals: list[float] = []
    meta: list[tuple[tuple[int, ...], str] | None] = []
    for leaf in raw_leaves:
        if leaf is None:
            meta.append(None)
            continue
        t = asarray(leaf)
        flat_vals.extend(_flatten_tensor_floats(t))
        meta.append((t.shape, t.dtype.value))

    n_params = len(flat_vals)
    flat_p_t = asarray(flat_vals)

    def _rebuild_from_tensor(vec_t: Tensor) -> Params:
        """Rebuild `Params` PyTree from 1D `Tensor` `vec_t`.

        Args:
            vec_t: 1D `Tensor` of parameter values.

        Returns:
            Reconstructed `Params` PyTree.
        """
        vec = _flatten_tensor_floats(vec_t)
        leaves_out: list[TensorLeaf] = []
        offset = 0
        for m in meta:
            if m is None:
                leaves_out.append(None)
                continue
            sh, dt_s = m
            cnt = 1
            for d in sh:
                cnt *= d
            sub = vec[offset : offset + cnt]
            offset += cnt
            if sh == ():
                leaves_out.append(asarray(sub[0], dtype=dt_s))
            else:
                leaves_out.append(reshape(asarray(sub, dtype=dt_s), sh))
        tree_out, _ = _unflatten_like(params, leaves_out, 0)
        return tree_out

    idx_vec = arange(n_params)
    grad_flat: list[float] = []
    for idx in range(n_params):
        basis = where(idx_vec == idx, eps, 0.0)
        f_plus = float(
            asarray(
                _call_value_fn(
                    value_fn, _rebuild_from_tensor(flat_p_t + basis), fn_args, kw
                )
            ).item()
        )
        f_minus = float(
            asarray(
                _call_value_fn(
                    value_fn, _rebuild_from_tensor(flat_p_t - basis), fn_args, kw
                )
            ).item()
        )
        grad_flat.append((f_plus - f_minus) / (2.0 * eps))

    return val_t, _rebuild_from_tensor(asarray(grad_flat))


def scale_by_backtracking_linesearch(
    max_backtracking_steps: int,
    slope_rtol: float = 0.0001,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
    store_grad: bool = False,
    verbose: bool = False,
) -> GradientTransformationExtraArgs[ScaleByBacktrackingLinesearchState]:
    """Backtracking line-search ensuring Armijo sufficient decrease.

    Args:
        max_backtracking_steps: Maximum number of backtracking iterations.
        slope_rtol: Relative tolerance on directional derivative slope (`c_1`).
        decrease_factor: Contraction factor (`< 1`) applied on rejected steps.
        increase_factor: Expansion factor (`>= 1`) applied to previous step size for initial guess.
        max_learning_rate: Maximum permissible learning rate.
        atol: Absolute tolerance for the Armijo decrease condition.
        rtol: Relative tolerance on `f(w)` for the Armijo decrease condition.
        store_grad: Whether to compute and store the gradient at the accepted iterate in state.
        verbose: Whether to print diagnostic messages on line-search failure.

    Returns:
        A `GradientTransformationExtraArgs` scaling updates by the line-search step size.
    """
    del verbose

    def init_fn(params: Params) -> ScaleByBacktrackingLinesearchState:
        """Initialize `ScaleByBacktrackingLinesearchState`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByBacktrackingLinesearchState`.
        """
        grad_init = tree_zeros_like(params) if store_grad else None
        return ScaleByBacktrackingLinesearchState(
            learning_rate=asarray(1.0),
            value=asarray(float("inf")),
            grad=grad_init,
            info=BacktrackingLinesearchInfo(
                num_linesearch_steps=asarray(0),
                decrease_error=asarray(float("inf")),
            ),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, ScaleByBacktrackingLinesearchState]:
        """Perform Armijo backtracking line search and scale `updates`.

        Args:
            updates: Search direction PyTree.
            state: Current `ScaleByBacktrackingLinesearchState`.
            params: Current model parameters PyTree (required).
            **extra_args: Must include `value_fn` and optionally `value`, `grad`, and data kwargs.

        Returns:
            Tuple of `(scaled_updates, new_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByBacktrackingLinesearchState`.
            ValueError: If `params` or `value_fn` is missing.
        """
        if params is None:
            raise ValueError(
                "`params` must be provided to `scale_by_backtracking_linesearch`."
            )
        if not isinstance(state, ScaleByBacktrackingLinesearchState):
            raise TypeError(
                f"Expected ScaleByBacktrackingLinesearchState, got {type(state)!r}."
            )
        raw_vfn = extra_args.get("value_fn")
        if not _is_value_fn(raw_vfn):
            raise ValueError(
                "`value_fn` callable must be passed to `scale_by_backtracking_linesearch.update`."
            )
        vfn: ValueFn = raw_vfn
        fn_kwargs = _filter_fn_kwargs(vfn, extra_args)

        raw_val = extra_args.get("value")
        raw_grad = extra_args.get("grad")
        if not _is_scalar_extra(raw_val) or raw_grad is None or callable(raw_grad):
            cur_val_t, cur_grad = _numerical_value_and_grad(vfn, params, (), fn_kwargs)
        else:
            cur_val_t = asarray(raw_val)
            cur_grad = raw_grad

        f0 = float(cur_val_t.item())
        slope = float(tree_vdot(updates, cur_grad).item())

        prev_lr = float(state.learning_rate.item())
        lr = min(increase_factor * prev_lr, max_learning_rate)
        dec_err = float("inf")
        new_val = f0
        cand_params: Params = params
        iter_num = 0

        while dec_err > atol and iter_num <= max_backtracking_steps:
            if iter_num > 0:
                lr = decrease_factor * lr
            cand_params = tree_add_scale(params, lr, updates) or params
            new_val = float(
                asarray(_call_value_fn(vfn, cand_params, (), fn_kwargs)).item()
            )
            raw_err = new_val - (1.0 + rtol) * f0 - lr * slope_rtol * slope
            dec_err = float("inf") if math.isnan(raw_err) else max(raw_err, 0.0)
            iter_num += 1

        final_lr = 0.0 if math.isinf(dec_err) else lr
        new_grad: Updates = None
        if store_grad:
            _, new_grad = _numerical_value_and_grad(vfn, cand_params, (), fn_kwargs)

        scaled_updates = tree_scalar_mul(final_lr, updates)
        new_state = ScaleByBacktrackingLinesearchState(
            learning_rate=asarray(final_lr),
            value=asarray(new_val),
            grad=new_grad,
            info=BacktrackingLinesearchInfo(
                num_linesearch_steps=asarray(iter_num),
                decrease_error=asarray(dec_err),
            ),
        )
        return scaled_updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def _interpolate_zoom_step(
    low: float,
    high: float,
    val_low: float,
    val_high: float,
    slope_low: float,
) -> float:
    """Compute safeguarded quadratic interpolation step in `[low, high]`.

    Args:
        low: Bracket endpoint with lower objective value.
        high: Other bracket endpoint.
        val_low: Objective value at `low`.
        val_high: Objective value at `high`.
        slope_low: Directional derivative at `low`.

    Returns:
        Safeguarded step size in `[min(low, high), max(low, high)]`.
    """
    diff = high - low
    denom = 2.0 * (val_high - val_low - slope_low * diff)
    lo_b = min(low, high)
    hi_b = max(low, high)
    width = hi_b - lo_b
    cand = (
        low - (slope_low * diff * diff) / denom
        if abs(denom) > 1e-12
        else 0.5 * (low + high)
    )
    margin = 0.1 * width
    if lo_b + margin <= cand <= hi_b - margin:
        return cand
    return 0.5 * (low + high)


def scale_by_zoom_linesearch(
    max_linesearch_steps: int,
    max_learning_rate: float | None = None,
    tol: float = 0.0,
    increase_factor: float = 2.0,
    slope_rtol: float = 0.0001,
    curv_rtol: float = 0.9,
    approx_dec_rtol: float | None = 1e-06,
    stepsize_precision: float = 1e-05,
    initial_guess_strategy: str = "keep",
    verbose: bool = False,
) -> GradientTransformationExtraArgs[ScaleByZoomLinesearchState]:
    """Zoom line-search ensuring Armijo sufficient decrease and Wolfe curvature conditions.

    Args:
        max_linesearch_steps: Maximum number of zoom line-search iterations.
        max_learning_rate: Optional upper bound on the step size.
        tol: Absolute tolerance for decrease and curvature conditions.
        increase_factor: Factor by which step size is increased during bracketing.
        slope_rtol: Relative slope tolerance `c_1` for Armijo decrease.
        curv_rtol: Relative curvature tolerance `c_2` for Wolfe curvature condition.
        approx_dec_rtol: Optional Hager-Zhang approximate decrease tolerance.
        stepsize_precision: Minimum bracket width before terminating bisection.
        initial_guess_strategy: Either `'keep'` (reuse previous step size) or `'one'` (`1.0`).
        verbose: Whether to print debugging information on failure.

    Returns:
        A `GradientTransformationExtraArgs` scaling updates by the zoom line-search step size.
    """
    del verbose

    def init_fn(params: Params) -> ScaleByZoomLinesearchState:
        """Initialize `ScaleByZoomLinesearchState`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByZoomLinesearchState`.
        """
        return ScaleByZoomLinesearchState(
            learning_rate=asarray(1.0),
            value=asarray(float("inf")),
            grad=tree_zeros_like(params),
            info=ZoomLinesearchInfo(
                num_linesearch_steps=asarray(0),
                decrease_error=asarray(float("inf")),
                curvature_error=asarray(float("inf")),
            ),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, ScaleByZoomLinesearchState]:
        """Run zoom/Wolfe line search along `updates` and return scaled updates.

        Args:
            updates: Search direction PyTree.
            state: Current `ScaleByZoomLinesearchState`.
            params: Current model parameters PyTree (required).
            **extra_args: Must include `value_fn` and optionally `value`, `grad`, and data kwargs.

        Returns:
            Tuple of `(scaled_updates, new_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByZoomLinesearchState`.
            ValueError: If `params` or `value_fn` is missing.
        """
        if params is None:
            raise ValueError("`params` must be provided to `scale_by_zoom_linesearch`.")
        if not isinstance(state, ScaleByZoomLinesearchState):
            raise TypeError(
                f"Expected ScaleByZoomLinesearchState, got {type(state)!r}."
            )
        raw_vfn = extra_args.get("value_fn")
        if not _is_value_fn(raw_vfn):
            raise ValueError(
                "`value_fn` callable must be passed to `scale_by_zoom_linesearch.update`."
            )
        vfn: ValueFn = raw_vfn
        fn_kwargs = _filter_fn_kwargs(vfn, extra_args)

        raw_val = extra_args.get("value")
        raw_grad = extra_args.get("grad")
        if not _is_scalar_extra(raw_val) or raw_grad is None or callable(raw_grad):
            cur_val_t, cur_grad = _numerical_value_and_grad(vfn, params, (), fn_kwargs)
        else:
            cur_val_t = asarray(raw_val)
            cur_grad = raw_grad

        f0 = float(cur_val_t.item())
        slope0 = float(tree_vdot(updates, cur_grad).item())

        lr = (
            1.0
            if initial_guess_strategy == "one"
            else float(state.learning_rate.item())
        )
        if max_learning_rate is not None:
            lr = min(lr, max_learning_rate)

        best_lr = lr
        best_val = f0
        best_grad = cur_grad
        dec_err = float("inf")
        curv_err = float("inf")
        low = 0.0
        val_low = f0
        slope_low = slope0
        high = float("inf")
        val_high = f0
        steps = 0

        for step_idx in range(max(max_linesearch_steps, 1)):
            steps = step_idx + 1
            cand_params = tree_add_scale(params, lr, updates)
            cand_val_t, cand_grad = _numerical_value_and_grad(
                vfn, cand_params, (), fn_kwargs
            )
            cand_val = float(cand_val_t.item())
            cand_slope = float(tree_vdot(updates, cand_grad).item())

            raw_dec = cand_val - (f0 + lr * slope_rtol * slope0)
            if approx_dec_rtol is not None:
                approx_dec = max(
                    cand_val - (f0 + approx_dec_rtol * abs(f0)),
                    cand_slope - (2.0 * slope_rtol - 1.0) * slope0,
                    0.0,
                )
                dec_err = min(max(raw_dec, 0.0), approx_dec)
            else:
                dec_err = max(raw_dec, 0.0)
            raw_curv = abs(cand_slope) - curv_rtol * abs(slope0)
            curv_err = max(raw_curv, 0.0)

            best_lr = lr
            best_val = cand_val
            best_grad = cand_grad

            if dec_err <= tol and curv_err <= tol:
                break
            if not math.isinf(high) and abs(high - low) <= stepsize_precision:
                break
            if dec_err > tol or (step_idx > 0 and cand_val >= val_low):
                high = lr
                val_high = cand_val
                lr = _interpolate_zoom_step(low, high, val_low, val_high, slope_low)
            elif cand_slope < 0.0 and math.isinf(high):
                low = lr
                val_low = cand_val
                slope_low = cand_slope
                lr = lr * increase_factor
                if max_learning_rate is not None:
                    lr = min(lr, max_learning_rate)
            else:
                low = lr
                val_low = cand_val
                slope_low = cand_slope
                lr = (
                    _interpolate_zoom_step(low, high, val_low, val_high, slope_low)
                    if not math.isinf(high)
                    else lr * 0.5
                )

        scaled_updates = tree_scalar_mul(best_lr, updates)
        new_state = ScaleByZoomLinesearchState(
            learning_rate=asarray(best_lr),
            value=asarray(best_val),
            grad=best_grad,
            info=ZoomLinesearchInfo(
                num_linesearch_steps=asarray(steps),
                decrease_error=asarray(dec_err),
                curvature_error=asarray(curv_err),
            ),
        )
        return scaled_updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def value_and_grad_from_state(
    value_fn: ValueFn,
) -> ValueAndGradFromStateFn:
    """Create a `(params, *args, state=..., **kwargs) -> (value, grad)` callable reusing cached state.

    Args:
        value_fn: Scalar objective function `value_fn(params, *args, **kwargs)`.

    Returns:
        Callable returning `(value, grad)` from `state` when valid and finite, or computing
        them via differentiation of `value_fn` otherwise.
    """

    def _value_and_grad(
        params: Params,
        *fn_args: PyTree,
        state: OptState,
        **fn_kwargs: PyTree,
    ) -> tuple[Tensor, Updates]:
        """Extract `(value, grad)` from `state` or evaluate `value_fn`.

        Args:
            params: Model parameters PyTree.
            *fn_args: Extra positional arguments forwarded to `value_fn`.
            state: Optimizer state containing `'value'` and `'grad'` fields.
            **fn_kwargs: Extra keyword arguments forwarded to `value_fn`.

        Returns:
            Tuple `(value, grad)`.

        Raises:
            ValueError: If `'value'` or `'grad'` is not found (or is None) in `state`.
        """
        val = tree_get(state, "value")
        grad = tree_get(state, "grad")
        if not _is_scalar_bound(val) or grad is None:
            raise ValueError(
                "Value or gradient not found in the state. "
                "Make sure that these values are stored in the state by the optimizer."
            )
        val_t = asarray(val)
        v_float = float(val_t.item())
        if not math.isinf(v_float) and not math.isnan(v_float):
            return val_t, grad
        return _numerical_value_and_grad(value_fn, params, fn_args, fn_kwargs)

    return _value_and_grad


__all__ = [
    "BacktrackingLinesearchInfo",
    "ScaleByBacktrackingLinesearchState",
    "ScaleByZoomLinesearchState",
    "ValueAndGradFromStateFn",
    "ValueFn",
    "ZoomLinesearchInfo",
    "_numerical_value_and_grad",
    "scale_by_backtracking_linesearch",
    "scale_by_zoom_linesearch",
    "value_and_grad_from_state",
]
