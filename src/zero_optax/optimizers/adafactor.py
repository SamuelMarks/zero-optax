"""Adafactor optimizer and factored second-moment transformations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from zero_optax._tree import (
    asarray,
    expand_dims,
    maximum,
    minimum,
    power,
    reduce_mean,
    sqrt,
    tree_map,
    zeros,
    zeros_like,
)
from zero_optax._types import (
    DTypeLike,
    MaskOrFn,
    Params,
    PyTree,
    ScalarOrSchedule,
    Tensor,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
    _as_count_tensor,
    _OptStateDataclassBase,
    identity,
)
from zero_optax.clipping import clip_by_block_rms
from zero_optax.combine import chain
from zero_optax.transform import (
    add_decayed_weights,
    scale_by_learning_rate,
    scale_by_param_block_rms,
    trace,
)


@dataclass(frozen=True)
class FactoredState(_OptStateDataclassBase):
    """State for `scale_by_factored_rms` holding step count and factored/unfactored moments.

    Attributes:
        count: Optimization step counter Tensor.
        v_row: Row second-moment factors PyTree.
        v_col: Column second-moment factors PyTree.
        v: Unfactored second-moment accumulators PyTree.
    """

    count: Tensor
    v_row: Updates
    v_col: Updates
    v: Updates
    _fields: tuple[str, ...] = ("count", "v_row", "v_col", "v")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'v_row'`, `'v_col'`, and `'v'`.
        """
        return {
            "count": self.count,
            "v_row": self.v_row,
            "v_col": self.v_col,
            "v": self.v,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> FactoredState:
        """Construct `FactoredState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, v_row, v_col, v)`.

        Returns:
            Reconstructed `FactoredState`.
        """
        items = list(iterable)
        return cls(
            count=_as_count_tensor(items[0]),
            v_row=items[1],
            v_col=items[2],
            v=items[3],
        )


def _default_decay_rate_fn(step: int, decay_rate: float) -> float:
    """Compute default Adafactor second-moment decay rate `1 - (step + 1) ** (-decay_rate)`.

    Args:
        step: Zero-based step index.
        decay_rate: Decay exponent.

    Returns:
        Scalar decay factor in `[0, 1)`.
    """
    step_t = maximum(asarray(step) + 1.0, 1.0)
    val_t = minimum(1.0 - power(step_t, -float(decay_rate)), 0.999999)
    return float(val_t.item())


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon: float = 1e-30,
    decay_rate_fn: Callable[[int, float], float] = _default_decay_rate_fn,
) -> GradientTransformation[FactoredState]:
    """Scale updates by a factored (or unfactored) root-mean-square second-moment estimate.

    Args:
        factored: Whether to use factored row/column second-moment estimates when dimensions allow.
        decay_rate: Exponent controlling the second-moment decay schedule `1 - t ** (-decay_rate)`.
        step_offset: Step offset subtracted/added to the step counter `t`.
        min_dim_size_to_factor: Minimum dimension size required on the last two axes to factor.
        epsilon: Regularization constant added to squared gradients.
        decay_rate_fn: Callable `(step, decay_rate) -> float` computing the decay rate at `step`.

    Returns:
        A `GradientTransformation` implementing Adafactor RMS scaling.
    """

    def _should_factor(shape: tuple[int, ...]) -> bool:
        """Check whether a tensor of `shape` should use factored second moments.

        Args:
            shape: Tensor shape tuple.

        Returns:
            True if `factored` is True, `len(shape) >= 2`, and the last two dims are `>= min_dim_size_to_factor`.
        """
        if not factored or len(shape) < 2:
            return False
        return (
            shape[-2] >= min_dim_size_to_factor and shape[-1] >= min_dim_size_to_factor
        )

    def init_fn(params: Params) -> FactoredState:
        """Initialize `FactoredState` for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `FactoredState`.
        """

        def _init_row(p: TensorLeaf) -> TensorLeaf:
            """Initialize `v_row` leaf if factored, else 1-element zero tensor.

            Args:
                p: Parameter leaf.

            Returns:
                Row factor Tensor leaf.
            """
            if p is None:
                return None
            p_t = asarray(p)
            if _should_factor(p_t.shape):
                return zeros(p_t.shape[:-1], dtype=p_t.dtype)
            return zeros((1,), dtype=p_t.dtype)

        def _init_col(p: TensorLeaf) -> TensorLeaf:
            """Initialize `v_col` leaf if factored, else 1-element zero tensor.

            Args:
                p: Parameter leaf.

            Returns:
                Column factor Tensor leaf.
            """
            if p is None:
                return None
            p_t = asarray(p)
            if _should_factor(p_t.shape):
                return zeros((*p_t.shape[:-2], p_t.shape[-1]), dtype=p_t.dtype)
            return zeros((1,), dtype=p_t.dtype)

        def _init_v(p: TensorLeaf) -> TensorLeaf:
            """Initialize `v` leaf to `zeros((1,))` when factored else `zeros_like(p)`.

            Args:
                p: Parameter leaf.

            Returns:
                Unfactored second-moment Tensor leaf.
            """
            if p is None:
                return None
            p_t = asarray(p)
            if _should_factor(p_t.shape):
                return zeros((1,), dtype=p_t.dtype)
            return zeros_like(p_t)

        return FactoredState(
            count=asarray(0),
            v_row=tree_map(_init_row, params),
            v_col=tree_map(_init_col, params),
            v=tree_map(_init_v, params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, FactoredState]:
        """Update factored/unfactored second moments and rescale `updates`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `FactoredState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(scaled_updates, new_factored_state)`.

        Raises:
            TypeError: If `state` is not a `FactoredState`.
        """
        del params
        if not isinstance(state, FactoredState):
            raise TypeError(
                f"Expected FactoredState in scale_by_factored_rms, got {type(state)!r}."
            )
        count_inc = state.count + 1
        step_shifted = int(maximum(state.count - step_offset, 0).item())
        decay_t = float(decay_rate_fn(step_shifted, decay_rate))

        def _update_v_row(vr: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update row second moment factor when `_should_factor(g.shape)`.

            Args:
                vr: Previous row factor leaf.
                g: Current gradient leaf.

            Returns:
                Updated row factor leaf.
            """
            if vr is None or g is None:
                return vr
            g_t = asarray(g)
            if not _should_factor(g_t.shape):
                return asarray(vr)
            g2 = g_t * g_t + epsilon
            row_mean = reduce_mean(g2, axis=-1, keepdims=False)
            return asarray(decay_t * asarray(vr) + (1.0 - decay_t) * row_mean)

        def _update_v_col(vc: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update column second moment factor when `_should_factor(g.shape)`.

            Args:
                vc: Previous column factor leaf.
                g: Current gradient leaf.

            Returns:
                Updated column factor leaf.
            """
            if vc is None or g is None:
                return vc
            g_t = asarray(g)
            if not _should_factor(g_t.shape):
                return asarray(vc)
            g2 = g_t * g_t + epsilon
            col_mean = reduce_mean(g2, axis=-2, keepdims=False)
            return asarray(decay_t * asarray(vc) + (1.0 - decay_t) * col_mean)

        def _update_v(v_leaf: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update unfactored second moment `decay_t * v + (1 - decay_t) * (g^2 + eps)`.

            Args:
                v_leaf: Previous second-moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second-moment leaf.
            """
            if v_leaf is None or g is None:
                return v_leaf
            g_t = asarray(g)
            if _should_factor(g_t.shape):
                return asarray(v_leaf)
            g2 = g_t * g_t + epsilon
            return asarray(decay_t * asarray(v_leaf) + (1.0 - decay_t) * g2)

        def _scale_leaf(
            g: TensorLeaf,
            vr_new: TensorLeaf,
            vc_new: TensorLeaf,
            v_new: TensorLeaf,
        ) -> TensorLeaf:
            """Scale `g` using reconstructed factored outer product or unfactored `v_new`.

            Args:
                g: Current gradient leaf.
                vr_new: Updated row factor leaf.
                vc_new: Updated column factor leaf.
                v_new: Updated unfactored second-moment leaf.

            Returns:
                Scaled gradient leaf.
            """
            if g is None or vr_new is None or vc_new is None or v_new is None:
                return g
            g_t = asarray(g)
            if _should_factor(g_t.shape):
                vr_t = asarray(vr_new)
                vc_t = asarray(vc_new)
                row_factor = expand_dims(vr_t, -1)
                col_factor = expand_dims(vc_t, -2)
                row_mean = expand_dims(reduce_mean(vr_t, axis=-1, keepdims=True), -1)
                r_factor = row_factor / row_mean
                c_factor = col_factor
                return g_t / (sqrt(r_factor) * sqrt(c_factor))
            return g_t / sqrt(asarray(v_new))

        new_v_row = tree_map(_update_v_row, state.v_row, updates)
        new_v_col = tree_map(_update_v_col, state.v_col, updates)
        new_v = tree_map(_update_v, state.v, updates)
        out_updates = tree_map(_scale_leaf, updates, new_v_row, new_v_col, new_v)
        return (
            out_updates,
            FactoredState(
                count=count_inc,
                v_row=new_v_row,
                v_col=new_v_col,
                v=new_v,
            ),
        )

    return GradientTransformation(init_fn, update_fn)


def adafactor(
    learning_rate: ScalarOrSchedule | None = None,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: float | None = 1.0,
    momentum: float | None = None,
    dtype_momentum: DTypeLike = "float32",
    weight_decay_rate: ScalarOrSchedule | None = None,
    eps: float = 1e-30,
    factored: bool = True,
    weight_decay_mask: MaskOrFn = None,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The Adafactor optimizer.

    Args:
        learning_rate: Optional global scaling factor or `Schedule`. If None, uses
            `min(1e-2, 1 / sqrt(t + 1))` when `multiply_by_parameter_scale=True` or
            `1 / sqrt(t + 1)` otherwise.
        min_dim_size_to_factor: Minimum dimension size required to factor second moments.
        decay_rate: Exponent controlling second-moment decay rate.
        decay_offset: Fine-tuning step offset for second-moment decay.
        multiply_by_parameter_scale: If True, multiply learning rate by parameter block RMS.
        clipping_threshold: Optional block RMS clipping threshold (`>= 1.0`), or None to disable.
        momentum: Optional momentum decay factor in `(0, 1)`.
        dtype_momentum: Data type for momentum buffers.
        weight_decay_rate: Optional weight decay rate or `Schedule`.
        eps: Regularization constant for squared gradients.
        factored: Whether to use factored second-moment estimates.
        weight_decay_mask: Optional mask PyTree or callable `mask(params)` for weight decay.

    Returns:
        A `GradientTransformation` implementing Adafactor.
    """

    def _default_adafactor_lr(step: int | Tensor) -> Tensor:
        """Compute default Adafactor step size schedule.

        Args:
            step: Optimization step count.

        Returns:
            Scalar step size Tensor.
        """
        step_t = asarray(step, dtype="float32")
        inv_sqrt = 1.0 / sqrt(step_t + 1.0)
        if multiply_by_parameter_scale:
            return minimum(inv_sqrt, 1e-2)
        return inv_sqrt * 1e-2

    lr_eff: ScalarOrSchedule = (
        learning_rate if learning_rate is not None else _default_adafactor_lr
    )
    clip_tx = (
        clip_by_block_rms(clipping_threshold)
        if clipping_threshold is not None
        else identity()
    )
    param_scale_tx = (
        scale_by_param_block_rms() if multiply_by_parameter_scale else identity()
    )
    mom_tx = (
        trace(decay=momentum, accumulator_dtype=dtype_momentum)
        if momentum is not None
        else identity()
    )
    wd_tx = (
        add_decayed_weights(weight_decay_rate, mask=weight_decay_mask)
        if weight_decay_rate is not None
        else identity()
    )
    return chain(
        scale_by_factored_rms(
            factored=factored,
            decay_rate=decay_rate,
            step_offset=decay_offset,
            min_dim_size_to_factor=min_dim_size_to_factor,
            epsilon=eps,
        ),
        clip_tx,
        scale_by_learning_rate(lr_eff),
        param_scale_tx,
        mom_tx,
        wd_tx,
    )


__all__ = [
    "FactoredState",
    "adafactor",
    "scale_by_factored_rms",
]
