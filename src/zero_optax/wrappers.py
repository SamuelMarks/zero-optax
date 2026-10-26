"""Optimizer wrappers (`masked`, `freeze`, `flatten`, `lookahead`, `MultiSteps`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Union

from zero_optax._tree import (
    INF,
    _coerce_pytree_child,
    _coerce_submask,
    _rebuild_mapping,
    _rebuild_sequence,
    abs_tensor,
    apply_mask_tree,
    asarray,
    reduce_sum,
    reshape,
    resolve_mask,
    tree_l2_norm,
    tree_leaves,
    tree_zeros_like,
)
from zero_optax._types import (
    ArrayLike,
    CustomMappingPyTreeProtocol,
    DTypeLike,
    MaskedNode,
    MaskOrFn,
    MaskTree,
    Params,
    PyTree,
    Tensor,
    TensorLeaf,
    Updates,
    VariableLeafProtocol,
)
from zero_optax.base import (
    ExtraArgValue,
    GradientTransformation,
    GradientTransformationExtraArgs,
    OptState,
    _as_count_tensor,
    _OptStateDataclassBase,
    set_to_zero,
)
from zero_optax.projections._projections import _is_scalar_bound
from zero_optax.tree_utils._random import _unflatten_like
from zero_optax.tree_utils._tree_math import (
    _flatten_tensor_floats,
    _structure_signature,
    tree_add,
    tree_scalar_mul,
    tree_sub,
)


class MaskedState(NamedTuple):
    """State for `masked` holding the inner transformation state on unmasked leaves.

    Attributes:
        inner_state: Inner optimizer state for `True` mask leaves.
    """

    inner_state: OptState


class LookaheadParams(NamedTuple):
    """Pair of `(fast, slow)` parameters used with `lookahead`.

    Attributes:
        fast: Fast parameters updated at every step.
        slow: Slow parameters synchronized every `sync_period` steps.
    """

    fast: Params
    slow: Params

    @classmethod
    def init_synced(cls, params: Params) -> LookaheadParams:
        """Create `LookaheadParams` with `fast` and `slow` initialized to `params`.

        Args:
            params: Initial model parameters PyTree.

        Returns:
            Synchronized `LookaheadParams(fast=params, slow=params)`.
        """
        return cls(fast=params, slow=params)


@dataclass(frozen=True)
class LookaheadState(_OptStateDataclassBase):
    """State for `lookahead` holding fast optimizer state, slow parameters, and step count.

    Attributes:
        fast_state: State of the wrapped `fast_optimizer`.
        slow_params: Slow parameter copy synchronized every `sync_period` steps.
        steps_since_sync: Step counter since the last synchronization.
    """

    fast_state: OptState
    slow_params: Params
    steps_since_sync: Tensor
    _fields: tuple[str, ...] = ("fast_state", "slow_params", "steps_since_sync")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'fast_state'`, `'slow_params'`, and `'steps_since_sync'`.
        """
        return {
            "fast_state": self.fast_state,
            "slow_params": self.slow_params,
            "steps_since_sync": self.steps_since_sync,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> LookaheadState:
        """Construct `LookaheadState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(fast_state, slow_params, steps_since_sync)`.

        Returns:
            Reconstructed `LookaheadState`.
        """
        items = list(iterable)
        return cls(
            fast_state=items[0],
            slow_params=items[1],
            steps_since_sync=_as_count_tensor(items[2]),
        )


ShouldSkipUpdateFunction = Callable[
    [Updates, Tensor, Union[Params, None]],
    tuple[Tensor, PyTree],
]


@dataclass(frozen=True)
class MultiStepsState(_OptStateDataclassBase):
    """State for `MultiSteps` gradient accumulation wrapper.

    Attributes:
        mini_step: Counter of mini-steps within the current accumulation cycle.
        gradient_step: Counter of completed outer optimizer updates.
        inner_opt_state: State of the wrapped inner optimizer.
        acc_grads: Accumulated gradients PyTree.
        skip_state: State managed by `should_skip_update_fn`.
    """

    mini_step: Tensor
    gradient_step: Tensor
    inner_opt_state: OptState
    acc_grads: Updates
    skip_state: PyTree = ()
    _fields: tuple[str, ...] = (
        "mini_step",
        "gradient_step",
        "inner_opt_state",
        "acc_grads",
        "skip_state",
    )

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary of `MultiStepsState` fields.
        """
        return {
            "mini_step": self.mini_step,
            "gradient_step": self.gradient_step,
            "inner_opt_state": self.inner_opt_state,
            "acc_grads": self.acc_grads,
            "skip_state": self.skip_state,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> MultiStepsState:
        """Construct `MultiStepsState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(mini_step, gradient_step, inner_opt_state, acc_grads, skip_state)`.

        Returns:
            Reconstructed `MultiStepsState`.
        """
        items = list(iterable)
        return cls(
            mini_step=_as_count_tensor(items[0]),
            gradient_step=_as_count_tensor(items[1]),
            inner_opt_state=items[2],
            acc_grads=items[3],
            skip_state=items[4] if len(items) > 4 else (),
        )


def skip_not_finite(
    updates: Updates,
    gradient_step: Tensor,
    params: Params | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Skip updates when any element in `updates` is `NaN` or `Inf`.

    Args:
        updates: Gradient updates PyTree.
        gradient_step: Current outer gradient step Tensor.
        params: Optional model parameters PyTree.

    Returns:
        Tuple `(should_skip, info_dict)`.
    """
    del gradient_step, params
    has_bad = False
    for leaf in tree_leaves(updates):
        if leaf is not None and not isinstance(leaf, MaskedNode):
            t = asarray(leaf)
            nan_cnt = int(reduce_sum(t != t).item())
            inf_cnt = int(reduce_sum(abs_tensor(t) == INF).item())
            if nan_cnt > 0 or inf_cnt > 0:
                has_bad = True
                break
    flag = asarray(has_bad)
    return flag, {"should_skip": flag}


def skip_large_deltas(
    updates: Updates,
    gradient_step: Tensor,
    params: Params | None,
    max_Sqr_norm: float = 1e4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Skip updates when the squared L2 norm of `updates` exceeds `max_Sqr_norm`.

    Args:
        updates: Gradient updates PyTree.
        gradient_step: Current outer gradient step Tensor.
        params: Optional model parameters PyTree.
        max_Sqr_norm: Maximum permissible squared L2 norm.

    Returns:
        Tuple `(should_skip, info_dict)`.
    """
    del gradient_step, params
    sq_sum_t = tree_l2_norm(updates, squared=True)
    nan_cnt = int(reduce_sum(sq_sum_t != sq_sum_t).item())
    inf_cnt = int(reduce_sum(abs_tensor(sq_sum_t) == INF).item())
    should_skip = nan_cnt > 0 or inf_cnt > 0 or (float(sq_sum_t.item()) >= max_Sqr_norm)
    flag = asarray(should_skip)
    return flag, {"should_skip": flag, "norm_sqr": sq_sum_t}


def _mask_tree_with_sentinel(tree: PyTree, mask_tree: MaskTree) -> PyTree:
    """Replace leaves/subtrees of `tree` where `mask_tree` is `False` with `MaskedNode()`.

    Args:
        tree: Input PyTree (`params` or `updates`).
        mask_tree: Boolean or nested boolean prefix PyTree.

    Returns:
        PyTree with `MaskedNode()` where `mask_tree` is `False`.
    """
    if isinstance(mask_tree, VariableLeafProtocol):
        return tree if bool(mask_tree.value) else MaskedNode()
    if isinstance(mask_tree, bool):
        return tree if mask_tree else MaskedNode()
    if isinstance(mask_tree, (Mapping, CustomMappingPyTreeProtocol)) and isinstance(
        tree, (Mapping, CustomMappingPyTreeProtocol)
    ):
        mask_keys = set(mask_tree.keys())
        return _rebuild_mapping(
            tree,
            {
                k: _mask_tree_with_sentinel(
                    _coerce_pytree_child(tree[k]),
                    _coerce_submask(_coerce_pytree_child(mask_tree[k]))
                    if k in mask_keys
                    else True,
                )
                for k in tree
            },
        )
    if (
        isinstance(mask_tree, (tuple, list))
        and isinstance(tree, (tuple, list))
        and not isinstance(tree, MaskedNode)
    ):
        return _rebuild_sequence(
            tree,
            [
                _mask_tree_with_sentinel(t_val, m_val)
                for t_val, m_val in zip(tree, mask_tree)
            ],
        )
    return tree


def _unmask_tree_with_fallback(
    masked_tree: PyTree,
    fallback_tree: PyTree,
    mask_tree: MaskTree,
) -> PyTree:
    """Restore leaves from `fallback_tree` wherever `mask_tree` is `False` or `masked_tree` is `MaskedNode`.

    Args:
        masked_tree: PyTree returned by `inner.update`.
        fallback_tree: Original unmasked `updates` PyTree.
        mask_tree: Resolved boolean prefix `MaskTree`.

    Returns:
        Unmasked PyTree combining transformed `True` leaves with original `False` leaves.
    """
    if isinstance(masked_tree, MaskedNode):
        return fallback_tree
    if isinstance(mask_tree, VariableLeafProtocol):
        return masked_tree if bool(mask_tree.value) else fallback_tree
    if isinstance(mask_tree, bool):
        return masked_tree if mask_tree else fallback_tree
    if (
        isinstance(mask_tree, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(masked_tree, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(fallback_tree, (Mapping, CustomMappingPyTreeProtocol))
    ):
        mask_keys = set(mask_tree.keys())
        return _rebuild_mapping(
            fallback_tree,
            {
                k: _unmask_tree_with_fallback(
                    _coerce_pytree_child(masked_tree[k]),
                    _coerce_pytree_child(fallback_tree[k]),
                    _coerce_submask(_coerce_pytree_child(mask_tree[k]))
                    if k in mask_keys
                    else True,
                )
                for k in fallback_tree
            },
        )
    if (
        isinstance(mask_tree, (tuple, list))
        and isinstance(masked_tree, (tuple, list))
        and isinstance(fallback_tree, (tuple, list))
    ):
        return _rebuild_sequence(
            fallback_tree,
            [
                _unmask_tree_with_fallback(m_val, f_val, b_val)
                for m_val, f_val, b_val in zip(masked_tree, fallback_tree, mask_tree)
            ],
        )
    return apply_mask_tree(masked_tree, fallback_tree, mask_tree)


def masked(
    inner: GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
    mask: MaskOrFn,
    *,
    mask_compatible_extra_args: bool = False,
) -> GradientTransformationExtraArgs[MaskedState]:
    """Mask updates so only `True` leaves are transformed by `inner` and `False` leaves pass through.

    Args:
        inner: Inner gradient transformation to apply on `True` mask leaves.
        mask: Boolean prefix PyTree or callable `mask(params) -> MaskTree`.
        mask_compatible_extra_args: Whether to also mask PyTree extra arguments with matching structure.

    Returns:
        A `GradientTransformationExtraArgs` wrapping `inner`.
    """

    def init_fn(params: Params) -> MaskedState:
        """Initialize `MaskedState` using `inner.init` on `MaskedNode`-masked `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `MaskedState`.
        """
        resolved_mask = resolve_mask(mask, params) if params is not None else None
        masked_params = (
            _mask_tree_with_sentinel(params, resolved_mask)
            if resolved_mask is not None
            else params
        )
        return MaskedState(inner_state=inner.init(masked_params))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, MaskedState]:
        """Apply `inner.update` on masked leaves and restore original updates where `mask` is `False`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `MaskedState`.
            params: Optional model parameters PyTree.
            **extra_args: Optional extra keyword arguments forwarded to `inner.update`.

        Returns:
            Tuple of `(masked_updates, MaskedState(new_inner_state))`.

        Raises:
            TypeError: If `state` is not a `MaskedState`.
        """
        if not isinstance(state, MaskedState):
            raise TypeError(f"Expected MaskedState in masked, got {type(state)!r}.")
        ref_tree = params if params is not None else updates
        resolved_mask: MaskTree | None = (
            resolve_mask(mask, ref_tree) if ref_tree is not None else None
        )
        masked_updates = (
            _mask_tree_with_sentinel(updates, resolved_mask)
            if resolved_mask is not None
            else updates
        )
        masked_params = (
            _mask_tree_with_sentinel(params, resolved_mask)
            if (params is not None and resolved_mask is not None)
            else params
        )
        forwarded_extra: dict[str, ExtraArgValue] = dict(extra_args)
        if mask_compatible_extra_args and resolved_mask is not None:
            upd_sig = _structure_signature(updates)
            for k_ex, v_ex in extra_args.items():
                if not callable(v_ex) and _structure_signature(v_ex) == upd_sig:
                    forwarded_extra[k_ex] = _mask_tree_with_sentinel(
                        v_ex, resolved_mask
                    )
        if isinstance(inner, GradientTransformationExtraArgs):
            new_updates, new_inner_state = inner.update(
                masked_updates, state.inner_state, masked_params, **forwarded_extra
            )
        else:
            new_updates, new_inner_state = inner.update(
                masked_updates, state.inner_state, masked_params
            )
        if resolved_mask is not None:
            out_updates = _unmask_tree_with_fallback(
                new_updates, updates, resolved_mask
            )
        else:
            out_updates = new_updates
        return out_updates, MaskedState(inner_state=new_inner_state)

    return GradientTransformationExtraArgs(init_fn, update_fn)


def freeze(mask: MaskOrFn) -> GradientTransformationExtraArgs[MaskedState]:
    """Create a transformation that zeros out gradient updates where `mask` is `True`.

    Args:
        mask: Boolean prefix PyTree or callable `mask(params) -> MaskTree` indicating frozen parameters.

    Returns:
        A `GradientTransformationExtraArgs` zeroing updates where `mask` is `True`.
    """
    return masked(set_to_zero(), mask)


def _flatten_pytree_to_1d(
    tree: PyTree,
) -> tuple[Tensor, list[tuple[tuple[int, ...], str] | None]]:
    """Flatten all leaves of `tree` into a single 1D `Tensor` and record leaf metadata.

    Args:
        tree: Input PyTree.

    Returns:
        Tuple `(flat_1d_tensor, leaf_metadata)`.
    """
    raw_leaves = tree_leaves(tree)
    flat_vals: list[float] = []
    meta: list[tuple[tuple[int, ...], str] | None] = []
    for leaf in raw_leaves:
        if leaf is None:
            meta.append(None)
            continue
        t = asarray(leaf)
        flat_vals.extend(_flatten_tensor_floats(t))
        meta.append((t.shape, t.dtype.value))
    return asarray(flat_vals), meta


def _unflatten_1d_to_pytree(
    flat_tensor: Tensor,
    template: PyTree,
    meta: Sequence[tuple[tuple[int, ...], str] | None],
) -> PyTree:
    """Unflatten a 1D `Tensor` back into the PyTree structure of `template`.

    Args:
        flat_tensor: 1D `Tensor` containing concatenated leaf values.
        template: Template PyTree defining container structure.
        meta: Leaf shape and dtype metadata from `_flatten_pytree_to_1d`.

    Returns:
        Reconstructed PyTree matching `template`.
    """
    vals = _flatten_tensor_floats(asarray(flat_tensor))
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
        sub = vals[offset : offset + cnt]
        offset += cnt
        if sh == ():
            leaves_out.append(asarray(sub[0], dtype=dt_s))
        else:
            leaves_out.append(reshape(asarray(sub, dtype=dt_s), sh))
    tree_out, _ = _unflatten_like(template, leaves_out, 0)
    return tree_out


def flatten(
    inner: GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
) -> GradientTransformationExtraArgs[OptState]:
    """Flatten parameters and gradients into a 1D vector for `inner.init` and `inner.update`.

    Args:
        inner: Inner transformation to wrap.

    Returns:
        A `GradientTransformationExtraArgs` flattening inputs to `inner`.
    """

    def init_fn(params: Params) -> OptState:
        """Flatten `params` to 1D and initialize `inner`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized state from `inner.init`.
        """
        flat_p, _ = _flatten_pytree_to_1d(params)
        return inner.init(flat_p)

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, OptState]:
        """Flatten `updates` (and `params`), apply `inner.update`, and unflatten updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current inner optimizer state.
            params: Optional model parameters PyTree.
            **extra_args: Optional extra keyword arguments forwarded to `inner.update`.

        Returns:
            Tuple of `(unflattened_updates, new_state)`.
        """
        flat_u, meta = _flatten_pytree_to_1d(updates)
        if flat_u.shape == (0,):
            return updates, state
        flat_p: Params | None = None
        if params is not None:
            flat_p, _ = _flatten_pytree_to_1d(params)
        if isinstance(inner, GradientTransformationExtraArgs):
            new_flat_u, new_state = inner.update(flat_u, state, flat_p, **extra_args)
        else:
            new_flat_u, new_state = inner.update(flat_u, state, flat_p)
        flat_res_t = asarray(new_flat_u) if _is_scalar_bound(new_flat_u) else flat_u
        unflat_u = _unflatten_1d_to_pytree(flat_res_t, updates, meta)
        return unflat_u, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def lookahead(
    fast_optimizer: GradientTransformation[OptState],
    sync_period: int,
    slow_step_size: float,
    reset_state: bool = False,
) -> GradientTransformation[LookaheadState]:
    """Lookahead optimizer synchronizing fast and slow parameters every `sync_period` steps.

    Args:
        fast_optimizer: Inner optimizer taking fast steps.
        sync_period: Number of fast steps between slow parameter synchronizations (`>= 1`).
        slow_step_size: Step size for the slow parameter interpolation in `[0, 1]`.
        reset_state: Whether to reset `fast_optimizer` state after each synchronization.

    Returns:
        A `GradientTransformation` implementing Lookahead.

    Raises:
        ValueError: If `sync_period < 1`.
    """
    if sync_period < 1:
        raise ValueError("Synchronization period must be >= 1.")

    def init_fn(params: Params) -> LookaheadState:
        """Initialize `LookaheadState` from `LookaheadParams` or standard `Params`.

        Args:
            params: `LookaheadParams(fast, slow)` or standard model parameters PyTree.

        Returns:
            Initialized `LookaheadState`.
        """
        fast_p = params.fast if isinstance(params, LookaheadParams) else params
        slow_p = params.slow if isinstance(params, LookaheadParams) else params
        return LookaheadState(
            fast_state=fast_optimizer.init(fast_p),
            slow_params=slow_p,
            steps_since_sync=asarray(0, dtype="int32"),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, LookaheadState]:
        """Update fast parameters and synchronize slow parameters every `sync_period` steps.

        Args:
            updates: Gradient updates computed with respect to fast parameters.
            state: Current `LookaheadState`.
            params: Current `LookaheadParams` or `Params` (required).

        Returns:
            Tuple of `(lookahead_updates, new_lookahead_state)`.

        Raises:
            TypeError: If `state` is not a `LookaheadState`.
            ValueError: If `params` is None.
        """
        if not isinstance(state, LookaheadState):
            raise TypeError(
                f"Expected LookaheadState in lookahead, got {type(state)!r}."
            )
        if params is None:
            raise ValueError("`params` must be provided to `lookahead` update.")
        fast_p = params.fast if isinstance(params, LookaheadParams) else params
        slow_p = (
            params.slow if isinstance(params, LookaheadParams) else state.slow_params
        )
        fast_updates, next_fast_state = fast_optimizer.update(
            updates, state.fast_state, fast_p
        )
        next_steps = int(state.steps_since_sync.item()) + 1
        is_sync = (next_steps % sync_period) == 0

        if is_sync:
            next_fast_params = tree_add(fast_p, fast_updates)
            slow_diff = tree_sub(next_fast_params, slow_p)
            slow_updates = tree_scalar_mul(slow_step_size, slow_diff)
            new_slow_params = tree_add(slow_p, slow_updates)
            synced_fast_updates = tree_sub(new_slow_params, fast_p)
            if reset_state:
                next_fast_state = fast_optimizer.init(new_slow_params)
            final_steps = 0
        else:
            synced_fast_updates = fast_updates
            slow_updates = tree_zeros_like(slow_p)
            new_slow_params = slow_p
            final_steps = next_steps

        out_updates: Updates = (
            LookaheadParams(fast=synced_fast_updates, slow=slow_updates)
            if isinstance(params, LookaheadParams)
            else synced_fast_updates
        )
        return (
            out_updates,
            LookaheadState(
                fast_state=next_fast_state,
                slow_params=new_slow_params,
                steps_since_sync=asarray(final_steps, dtype="int32"),
            ),
        )

    return GradientTransformation(init_fn, update_fn)


class MultiSteps:
    """An optimizer wrapper to accumulate gradients over multiple mini-steps."""

    _opt: GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState]
    _every_k_schedule: Callable[[ArrayLike], Tensor]
    _use_grad_mean: bool
    _should_skip_update_fn: ShouldSkipUpdateFunction
    _acc_dtype: DTypeLike

    def __init__(
        self,
        opt: GradientTransformation[OptState]
        | GradientTransformationExtraArgs[OptState],
        every_k_schedule: int | Callable[[ArrayLike], ArrayLike],
        use_grad_mean: bool = True,
        should_skip_update_fn: ShouldSkipUpdateFunction | None = None,
        accumulator_dtype: DTypeLike = "float32",
    ) -> None:
        """Initialize a `MultiSteps` gradient accumulation wrapper.

        Args:
            opt: Inner gradient transformation to apply once every `k` mini-steps.
            every_k_schedule: Integer `k` or schedule callable `k(gradient_step)` specifying
                how many mini-steps to accumulate per outer step.
            use_grad_mean: If True, average accumulated gradients over `k` steps; otherwise sum.
            should_skip_update_fn: Optional predicate `(updates, gradient_step, params) -> (should_skip, info)`
                to skip invalid outer updates.
            accumulator_dtype: Optional dtype for the gradient accumulator buffer.
        """
        self._opt = opt
        if callable(every_k_schedule):
            sched_fn = every_k_schedule

            def _sched_wrap(step: ArrayLike) -> Tensor:
                """Evaluate callable `every_k_schedule(step)` as a Tensor.

                Args:
                    step: Outer gradient step count.

                Returns:
                    Scheduled accumulation count Tensor.
                """
                return asarray(sched_fn(step), dtype="int32")

            self._every_k_schedule = _sched_wrap
        else:
            k_const = int(every_k_schedule)

            def _const_wrap(step: ArrayLike) -> Tensor:
                """Return constant `k` as an integer Tensor.

                Args:
                    step: Outer gradient step count (unused).

                Returns:
                    Constant accumulation count Tensor.
                """
                del step
                return asarray(k_const, dtype="int32")

            self._every_k_schedule = _const_wrap

        self._use_grad_mean = use_grad_mean
        self._acc_dtype = accumulator_dtype

        if should_skip_update_fn is None:

            def _never_skip(
                u: Updates,
                g_step: Tensor,
                p: Params | None,
            ) -> tuple[Tensor, tuple[()]]:
                """Default skip predicate that never skips updates.

                Args:
                    u: Updates PyTree.
                    g_step: Gradient step Tensor.
                    p: Optional parameters PyTree.

                Returns:
                    Tuple `(False, ())`.
                """
                del u, g_step, p
                return asarray(False), ()

            self._should_skip_update_fn = _never_skip
        else:
            self._should_skip_update_fn = should_skip_update_fn

    @property
    def inner_opt(
        self,
    ) -> GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState]:
        """Return the wrapped inner `GradientTransformation`.

        Returns:
            Wrapped `GradientTransformation`.
        """
        return self._opt

    def init(self, params: Params) -> MultiStepsState:
        """Initialize `MultiStepsState` from `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `MultiStepsState`.
        """
        zero_grads = tree_zeros_like(params, dtype=self._acc_dtype)
        _, skip_state = self._should_skip_update_fn(
            zero_grads, asarray(0, dtype="int32"), params
        )
        return MultiStepsState(
            mini_step=asarray(0, dtype="int32"),
            gradient_step=asarray(0, dtype="int32"),
            inner_opt_state=self._opt.init(params),
            acc_grads=zero_grads,
            skip_state=skip_state,
        )

    def update(
        self,
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, MultiStepsState]:
        """Accumulate `updates` and emit an outer update every `k` mini-steps.

        Args:
            updates: Current mini-step gradient updates PyTree.
            state: Current `MultiStepsState`.
            params: Optional model parameters PyTree.
            **extra_args: Optional extra keyword arguments forwarded to `self._opt.update`.

        Returns:
            Tuple of `(emitted_updates, new_multi_steps_state)`.

        Raises:
            TypeError: If `state` is not a `MultiStepsState`.
        """
        if not isinstance(state, MultiStepsState):
            raise TypeError(
                f"Expected MultiStepsState in MultiSteps.update, got {type(state)!r}."
            )
        k_steps = max(int(self._every_k_schedule(state.gradient_step).item()), 1)
        mini_step = int(state.mini_step.item())
        grad_step = int(state.gradient_step.item())

        if self._use_grad_mean:
            next_acc = tree_add(
                state.acc_grads,
                tree_scalar_mul(
                    1.0 / float(mini_step + 1),
                    tree_sub(updates, state.acc_grads),
                ),
            )
        else:
            next_acc = tree_add(state.acc_grads, updates)

        if (mini_step + 1) < k_steps:
            return (
                tree_zeros_like(updates),
                MultiStepsState(
                    mini_step=asarray(mini_step + 1, dtype="int32"),
                    gradient_step=asarray(grad_step, dtype="int32"),
                    inner_opt_state=state.inner_opt_state,
                    acc_grads=next_acc,
                    skip_state=state.skip_state,
                ),
            )

        should_skip_t, next_skip_state = self._should_skip_update_fn(
            next_acc, state.gradient_step, params
        )
        if bool(should_skip_t.item()):
            return (
                tree_zeros_like(updates),
                MultiStepsState(
                    mini_step=asarray(0, dtype="int32"),
                    gradient_step=asarray(grad_step, dtype="int32"),
                    inner_opt_state=state.inner_opt_state,
                    acc_grads=tree_zeros_like(updates, dtype=self._acc_dtype),
                    skip_state=next_skip_state,
                ),
            )

        if isinstance(self._opt, GradientTransformationExtraArgs):
            final_updates, next_inner_state = self._opt.update(
                next_acc, state.inner_opt_state, params, **extra_args
            )
        else:
            final_updates, next_inner_state = self._opt.update(
                next_acc, state.inner_opt_state, params
            )
        return (
            final_updates,
            MultiStepsState(
                mini_step=asarray(0, dtype="int32"),
                gradient_step=asarray(grad_step + 1, dtype="int32"),
                inner_opt_state=next_inner_state,
                acc_grads=tree_zeros_like(updates, dtype=self._acc_dtype),
                skip_state=next_skip_state,
            ),
        )

    def has_updated(self, state: MultiStepsState | PyTree) -> Tensor:
        """Check whether the outer optimizer updated parameters on the most recent step.

        Args:
            state: Current `MultiStepsState`.

        Returns:
            Scalar boolean Tensor (`True` if an outer step just occurred).

        Raises:
            TypeError: If `state` is not a `MultiStepsState`.
        """
        if not isinstance(state, MultiStepsState):
            raise TypeError(
                f"Expected MultiStepsState in has_updated, got {type(state)!r}."
            )
        mini = int(state.mini_step.item())
        g_step = int(state.gradient_step.item())
        return asarray(mini == 0 and g_step > 0)

    def gradient_transformation(
        self,
    ) -> GradientTransformationExtraArgs[MultiStepsState]:
        """Return a `GradientTransformationExtraArgs` view of this `MultiSteps` instance.

        Returns:
            `GradientTransformationExtraArgs(self.init, self.update)`.
        """
        return GradientTransformationExtraArgs(self.init, self.update)


__all__ = [
    "LookaheadParams",
    "LookaheadState",
    "MaskedNode",
    "MaskedState",
    "MultiSteps",
    "MultiStepsState",
    "ShouldSkipUpdateFunction",
    "flatten",
    "freeze",
    "lookahead",
    "masked",
    "skip_large_deltas",
    "skip_not_finite",
]
