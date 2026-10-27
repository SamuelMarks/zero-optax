"""Core composable gradient transformations (`scale_by_*`, `trace`, `add_decayed_weights`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

from zero_optax._tree import (
    abs_tensor,
    apply_mask_tree,
    asarray,
    evaluate_schedule,
    full_like,
    leaf_l2_norm,
    maximum,
    minimum,
    power,
    reduce_mean,
    reduce_sum,
    resolve_mask,
    safe_norm,
    sigmoid,
    sign,
    sqrt,
    tree_map,
    tree_zeros_like,
    where,
)
from zero_optax._types import (
    DTypeLike,
    MaskOrFn,
    Params,
    PyTree,
    ScalarOrSchedule,
    Schedule,
    Tensor,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    EmptyState,
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
    ScaleState,
    TraceState,
    _as_count_tensor,
    _OptStateDataclassBase,
    identity,
)
from zero_optax.tree_utils._random import (
    RngKeyLike,
    _extract_seed_int,
    _split_seed_to_keys,
    tree_random_like,
)


class ScaleByRssState(NamedTuple):
    """State holding the sum of squared gradients for `scale_by_rss` / `adagrad`.

    Attributes:
        sum_of_squares: PyTree of accumulated squared gradients.
    """

    sum_of_squares: Updates


@dataclass(frozen=True)
class ScaleByAdamState(_OptStateDataclassBase):
    """State holding step count and first/second moment estimates for Adam/Yogi.

    Attributes:
        count: Step counter Tensor.
        mu: First moment (mean) PyTree.
        nu: Second moment (uncentered variance) PyTree.
    """

    count: Tensor
    mu: Updates
    nu: Updates
    _fields: tuple[str, ...] = ("count", "mu", "nu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'mu'`, and `'nu'`.
        """
        return {"count": self.count, "mu": self.mu, "nu": self.nu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByAdamState:
        """Construct `ScaleByAdamState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu)`.

        Returns:
            Reconstructed `ScaleByAdamState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), mu=items[1], nu=items[2])


@dataclass(frozen=True)
class ScaleByBeliefState(_OptStateDataclassBase):
    """State holding step count, first moment, and central second moment for AdaBelief.

    Attributes:
        count: Step counter Tensor.
        mu: First moment PyTree.
        nu: Second central moment PyTree.
    """

    count: Tensor
    mu: Updates
    nu: Updates
    _fields: tuple[str, ...] = ("count", "mu", "nu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'mu'`, and `'nu'`.
        """
        return {"count": self.count, "mu": self.mu, "nu": self.nu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByBeliefState:
        """Construct `ScaleByBeliefState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu)`.

        Returns:
            Reconstructed `ScaleByBeliefState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), mu=items[1], nu=items[2])


@dataclass(frozen=True)
class ScaleByLionState(_OptStateDataclassBase):
    """State holding momentum estimate for Lion optimizer.

    Attributes:
        count: Step counter Tensor.
        mu: Momentum trace PyTree.
    """

    count: Tensor
    mu: Updates
    _fields: tuple[str, ...] = ("count", "mu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'` and `'mu'`.
        """
        return {"count": self.count, "mu": self.mu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByLionState:
        """Construct `ScaleByLionState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu)`.

        Returns:
            Reconstructed `ScaleByLionState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), mu=items[1])


@dataclass(frozen=True)
class ScaleByAdanState(_OptStateDataclassBase):
    """State holding first, second, difference moments, previous gradient, and step count for Adan.

    Attributes:
        m: First moment estimate PyTree.
        v: Gradient difference moment estimate PyTree.
        n: Second moment estimate PyTree.
        g: Previous step gradient PyTree.
        t: Step counter Tensor.
    """

    m: Updates
    v: Updates
    n: Updates
    g: Updates
    t: Tensor
    _fields: tuple[str, ...] = ("m", "v", "n", "g", "t")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'m'`, `'v'`, `'n'`, `'g'`, and `'t'`.
        """
        return {
            "m": self.m,
            "v": self.v,
            "n": self.n,
            "g": self.g,
            "t": self.t,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByAdanState:
        """Construct `ScaleByAdanState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(m, v, n, g, t)`.

        Returns:
            Reconstructed `ScaleByAdanState`.
        """
        items = list(iterable)
        return cls(
            m=items[0],
            v=items[1],
            n=items[2],
            g=items[3],
            t=_as_count_tensor(items[4]),
        )


class ScaleByRmsState(NamedTuple):
    """State holding exponentially weighted squared gradients for RMSProp.

    Attributes:
        nu: Second moment PyTree.
    """

    nu: Updates


@dataclass(frozen=True)
class ScaleByRmsWithCountState(_OptStateDataclassBase):
    """State holding step count and second moment for bias-corrected RMSProp.

    Attributes:
        count: Step counter Tensor.
        nu: Second moment PyTree.
    """

    count: Tensor
    nu: Updates
    _fields: tuple[str, ...] = ("count", "nu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'` and `'nu'`.
        """
        return {"count": self.count, "nu": self.nu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByRmsWithCountState:
        """Construct `ScaleByRmsWithCountState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, nu)`.

        Returns:
            Reconstructed `ScaleByRmsWithCountState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), nu=items[1])


class ScaleByRStdDevState(NamedTuple):
    """State holding first and second moments for centered RMSProp (`scale_by_stddev`).

    Attributes:
        mu: First moment PyTree.
        nu: Second moment PyTree.
    """

    mu: Updates
    nu: Updates


@dataclass(frozen=True)
class ScaleByRStdDevWithCountState(_OptStateDataclassBase):
    """State holding step count and first/second moments for bias-corrected centered RMSProp.

    Attributes:
        count: Step counter Tensor.
        mu: First moment PyTree.
        nu: Second moment PyTree.
    """

    count: Tensor
    mu: Updates
    nu: Updates
    _fields: tuple[str, ...] = ("count", "mu", "nu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'mu'`, and `'nu'`.
        """
        return {"count": self.count, "mu": self.mu, "nu": self.nu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByRStdDevWithCountState:
        """Construct `ScaleByRStdDevWithCountState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu)`.

        Returns:
            Reconstructed `ScaleByRStdDevWithCountState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), mu=items[1], nu=items[2])


@dataclass(frozen=True)
class ScaleByNovogradState(_OptStateDataclassBase):
    """State holding step count, first moment, and per-leaf second moment for NovoGrad.

    Attributes:
        count: Step counter Tensor.
        mu: First moment PyTree.
        nu: Per-leaf scalar second moment PyTree.
    """

    count: Tensor
    mu: Updates
    nu: Updates
    _fields: tuple[str, ...] = ("count", "mu", "nu")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'mu'`, and `'nu'`.
        """
        return {"count": self.count, "mu": self.mu, "nu": self.nu}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByNovogradState:
        """Construct `ScaleByNovogradState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu)`.

        Returns:
            Reconstructed `ScaleByNovogradState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), mu=items[1], nu=items[2])


@dataclass(frozen=True)
class AddDecayedWeightsState(_OptStateDataclassBase):
    """State holding optional step count for `add_decayed_weights`.

    Attributes:
        count: Step counter Tensor.
    """

    count: Tensor
    _fields: tuple[str, ...] = ("count",)

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`.
        """
        return {"count": self.count}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> AddDecayedWeightsState:
        """Construct `AddDecayedWeightsState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count,)`.

        Returns:
            Reconstructed `AddDecayedWeightsState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]))


class ScaleByTrustRatioState(NamedTuple):
    """Empty state for `scale_by_trust_ratio`."""


def scale(step_size: float | int | Tensor) -> GradientTransformation[ScaleState]:
    """Scale gradient updates by a constant factor `step_size`.

    Args:
        step_size: Constant scalar multiplier applied to all updates.

    Returns:
        A `GradientTransformation` multiplying updates by `step_size`.
    """

    def init_fn(params: Params) -> ScaleState:
        """Initialize empty `ScaleState`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `ScaleState`.
        """
        del params
        return ScaleState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleState]:
        """Multiply every leaf in `updates` by `step_size`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(scaled_updates, ScaleState())`.
        """
        del state, params

        def _mul(u: TensorLeaf) -> TensorLeaf:
            """Multiply leaf `u` by `step_size`.

            Args:
                u: Gradient update leaf.

            Returns:
                Scaled Tensor leaf.
            """
            return None if u is None else step_size * asarray(u)

        return tree_map(_mul, updates), ScaleState()

    return GradientTransformation(init_fn, update_fn)


def scale_by_schedule(
    step_size_fn: Schedule,
) -> GradientTransformation[ScaleByScheduleState]:
    """Scale gradient updates using a time-varying schedule function `step_size_fn`.

    Args:
        step_size_fn: Schedule callable mapping step `count` to a scalar multiplier.

    Returns:
        A `GradientTransformation` scaling updates by `step_size_fn(count)`.
    """

    def init_fn(params: Params) -> ScaleByScheduleState:
        """Initialize `ScaleByScheduleState` with `count = 0`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `ScaleByScheduleState`.
        """
        del params
        return ScaleByScheduleState(count=asarray(0))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByScheduleState]:
        """Scale updates by `step_size_fn(state.count)` and increment `count`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByScheduleState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(scaled_updates, new_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByScheduleState`.
        """
        del params
        if not isinstance(state, ScaleByScheduleState):
            raise TypeError(
                f"Expected ScaleByScheduleState in scale_by_schedule, got {type(state)!r}."
            )
        count = state.count
        step_size = step_size_fn(count)

        def _scale_leaf(u: TensorLeaf) -> TensorLeaf:
            """Multiply leaf `u` by `step_size`.

            Args:
                u: Gradient update leaf.

            Returns:
                Scaled Tensor leaf.
            """
            return None if u is None else step_size * asarray(u)

        return tree_map(_scale_leaf, updates), ScaleByScheduleState(count=count + 1)

    return GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(
    learning_rate: ScalarOrSchedule | None = None,
    *,
    flip_sign: bool = True,
) -> GradientTransformation[ScaleByScheduleState | ScaleState | EmptyState]:
    """Scale updates by `learning_rate` (negated for gradient descent when `flip_sign=True`).

    Args:
        learning_rate: Fixed scalar learning rate, callable `Schedule`, or None for no scaling.
        flip_sign: If True, multiply by `-learning_rate`; otherwise `+learning_rate`.

    Returns:
        A `GradientTransformation` scaling updates by the learning rate.
    """
    if learning_rate is None:
        return identity()
    sign_mult = -1.0 if flip_sign else 1.0
    if callable(learning_rate):

        def _sched(count: int | Tensor) -> float | Tensor:
            """Evaluate scheduled learning rate with sign flip.

            Args:
                count: Optimization step count.

            Returns:
                Signed step size.
            """
            return sign_mult * learning_rate(count)

        return scale_by_schedule(_sched)
    return scale(sign_mult * learning_rate)


def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: DTypeLike = None,
) -> GradientTransformation[TraceState]:
    """Compute a trace (momentum) of past gradient updates.

    Args:
        decay: Exponential decay factor for the trace.
        nesterov: Whether to apply Nesterov momentum lookahead.
        accumulator_dtype: Optional data type for the trace accumulator.

    Returns:
        A `GradientTransformation` applying classical or Nesterov momentum.
    """

    def init_fn(params: Params) -> TraceState:
        """Initialize `TraceState` with zeros matching `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `TraceState`.
        """
        return TraceState(trace=tree_zeros_like(params, dtype=accumulator_dtype))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, TraceState]:
        """Update momentum trace and compute output updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `TraceState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(momentum_updates, new_trace_state)`.

        Raises:
            TypeError: If `state` is not a `TraceState`.
        """
        del params
        if not isinstance(state, TraceState):
            raise TypeError(f"Expected TraceState in trace, got {type(state)!r}.")
        prev_trace = state.trace

        def _update_trace(g: TensorLeaf, t: TensorLeaf) -> TensorLeaf:
            """Compute new trace `g + decay * t`.

            Args:
                g: Current gradient leaf.
                t: Previous trace leaf.

            Returns:
                Updated trace Tensor leaf.
            """
            if g is None or t is None:
                return g
            res = asarray(g) + decay * asarray(t)
            return (
                asarray(res, dtype=accumulator_dtype)
                if accumulator_dtype is not None
                else res
            )

        new_trace = tree_map(_update_trace, updates, prev_trace)
        if nesterov:

            def _nesterov_step(g: TensorLeaf, t: TensorLeaf) -> TensorLeaf:
                """Compute Nesterov lookahead `g + decay * new_trace`.

                Args:
                    g: Current gradient leaf.
                    t: Updated trace leaf.

                Returns:
                    Nesterov update Tensor leaf.
                """
                if g is None or t is None:
                    return g
                return asarray(g) + decay * asarray(t)

            out_updates = tree_map(_nesterov_step, updates, new_trace)
        else:
            out_updates = new_trace
        return out_updates, TraceState(trace=new_trace)

    return GradientTransformation(init_fn, update_fn)


def scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> GradientTransformation[ScaleByRssState]:
    """Rescale updates by the root of the sum of all squared gradients to date.

    Args:
        initial_accumulator_value: Initial value for the sum-of-squares accumulator.
        eps: Small constant added inside the square root for numerical stability.

    Returns:
        A `GradientTransformation` implementing Adagrad's RSS rescaling.
    """

    def init_fn(params: Params) -> ScaleByRssState:
        """Initialize `ScaleByRssState` filled with `initial_accumulator_value`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByRssState`.
        """

        def _init_leaf(p: TensorLeaf) -> TensorLeaf:
            """Fill leaf accumulator with `initial_accumulator_value`.

            Args:
                p: Parameter leaf.

            Returns:
                Initialized accumulator Tensor leaf.
            """
            return None if p is None else full_like(p, initial_accumulator_value)

        return ScaleByRssState(sum_of_squares=tree_map(_init_leaf, params))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByRssState]:
        """Accumulate squared gradients and rescale updates by `1 / sqrt(rss + eps)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByRssState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(rescaled_updates, new_rss_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByRssState`.
        """
        del params
        if not isinstance(state, ScaleByRssState):
            raise TypeError(
                f"Expected ScaleByRssState in scale_by_rss, got {type(state)!r}."
            )
        prev_rss = state.sum_of_squares

        def _add_sq(rss: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Add `g^2` to `rss`.

            Args:
                rss: Previous sum of squares leaf.
                g: Current gradient leaf.

            Returns:
                Updated sum of squares leaf.
            """
            if rss is None or g is None:
                return rss
            g_t = asarray(g)
            return asarray(rss) + g_t * g_t

        new_rss = tree_map(_add_sq, prev_rss, updates)

        def _scale(rss: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Divide `g` by `sqrt(rss + eps)`.

            Args:
                rss: Updated sum of squares leaf.
                g: Current gradient leaf.

            Returns:
                Rescaled update leaf.
            """
            if rss is None or g is None:
                return g
            return asarray(g) / sqrt(asarray(rss) + eps)

        return tree_map(_scale, new_rss, updates), ScaleByRssState(
            sum_of_squares=new_rss
        )

    return GradientTransformation(init_fn, update_fn)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: DTypeLike = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation[ScaleByAdamState]:
    """Rescale updates according to the Adam algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        mu_dtype: Optional data type for storing the first moment accumulator.
        nesterov: Whether to use Nesterov momentum for the first moment estimate.

    Returns:
        A `GradientTransformation` implementing Adam moment rescaling.
    """

    def init_fn(params: Params) -> ScaleByAdamState:
        """Initialize `ScaleByAdamState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByAdamState`.
        """
        return ScaleByAdamState(
            count=asarray(0),
            mu=tree_zeros_like(params, dtype=mu_dtype),
            nu=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByAdamState]:
        """Update Adam moments and compute bias-corrected rescaled updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByAdamState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(adam_updates, new_adam_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByAdamState`.
        """
        del params
        if not isinstance(state, ScaleByAdamState):
            raise TypeError(
                f"Expected ScaleByAdamState in scale_by_adam, got {type(state)!r}."
            )
        count_inc = state.count + 1

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update first moment `b1 * m + (1 - b1) * g`.

            Args:
                m: Previous first moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated first moment leaf.
            """
            if m is None or g is None:
                return m
            res = b1 * asarray(m) + (1.0 - b1) * asarray(g)
            return asarray(res, dtype=mu_dtype) if mu_dtype is not None else res

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update second moment `b2 * v + (1 - b2) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            return b2 * asarray(v) + (1.0 - b2) * (g_t * g_t)

        new_mu = tree_map(_update_mu, state.mu, updates)
        new_nu = tree_map(_update_nu, state.nu, updates)

        bc1 = 1.0 - (b1**count_inc)
        bc2 = 1.0 - (b2**count_inc)
        bc1_next = 1.0 - (b1 ** (count_inc + 1))

        def _bias_correct_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Compute bias-corrected first moment (with optional Nesterov lookahead).

            Args:
                m: Updated first moment leaf.
                g: Current gradient leaf.

            Returns:
                Bias-corrected first moment leaf.
            """
            if m is None or g is None:
                return m
            if nesterov:
                return (b1 * asarray(m) / bc1_next) + ((1.0 - b1) * asarray(g) / bc1)
            return asarray(m) / bc1

        mu_hat = tree_map(_bias_correct_mu, new_mu, updates)

        def _scale_adam(m_hat: TensorLeaf, v: TensorLeaf) -> TensorLeaf:
            """Compute `m_hat / (sqrt(v / bc2 + eps_root) + eps)`.

            Args:
                m_hat: Bias-corrected first moment leaf.
                v: Updated second moment leaf.

            Returns:
                Rescaled Adam update leaf.
            """
            if m_hat is None or v is None:
                return m_hat
            v_hat = asarray(v) / bc2
            return asarray(m_hat) / (sqrt(v_hat + eps_root) + eps)

        out_updates = tree_map(_scale_adam, mu_hat, new_nu)
        return out_updates, ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    *,
    nesterov: bool = False,
) -> GradientTransformation[ScaleByBeliefState]:
    """Rescale updates according to the AdaBelief algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second central moment (variance) estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added to the prediction error variance at each step.
        nesterov: Whether to apply Nesterov momentum to the first moment estimate.

    Returns:
        A `GradientTransformation` implementing AdaBelief rescaling.
    """

    def init_fn(params: Params) -> ScaleByBeliefState:
        """Initialize `ScaleByBeliefState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByBeliefState`.
        """
        return ScaleByBeliefState(
            count=asarray(0),
            mu=tree_zeros_like(params),
            nu=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByBeliefState]:
        """Update AdaBelief moments and compute rescaled updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByBeliefState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(belief_updates, new_belief_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByBeliefState`.
        """
        del params
        if not isinstance(state, ScaleByBeliefState):
            raise TypeError(
                f"Expected ScaleByBeliefState in scale_by_belief, got {type(state)!r}."
            )
        count_inc = state.count + 1

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update first moment `b1 * m + (1 - b1) * g`.

            Args:
                m: Previous first moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated first moment leaf.
            """
            if m is None or g is None:
                return m
            return b1 * asarray(m) + (1.0 - b1) * asarray(g)

        new_mu = tree_map(_update_mu, state.mu, updates)

        def _update_nu(v: TensorLeaf, g: TensorLeaf, m: TensorLeaf) -> TensorLeaf:
            """Update central second moment `b2 * v + (1 - b2) * (g - m)^2 + eps_root`.

            Args:
                v: Previous central second moment leaf.
                g: Current gradient leaf.
                m: Updated first moment leaf.

            Returns:
                Updated central second moment leaf.
            """
            if v is None or g is None or m is None:
                return v
            diff = asarray(g) - asarray(m)
            return b2 * asarray(v) + (1.0 - b2) * (diff * diff) + eps_root

        new_nu = tree_map(_update_nu, state.nu, updates, new_mu)

        bc1 = 1.0 - (b1**count_inc)
        bc2 = 1.0 - (b2**count_inc)
        bc1_next = 1.0 - (b1 ** (count_inc + 1))

        def _compute_update(m: TensorLeaf, v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Compute bias-corrected AdaBelief update leaf.

            Args:
                m: Updated first moment leaf.
                v: Updated central second moment leaf.
                g: Current gradient leaf.

            Returns:
                Rescaled AdaBelief update leaf.
            """
            if m is None or v is None or g is None:
                return m
            if nesterov:
                m_hat = (b1 * asarray(m) / bc1_next) + ((1.0 - b1) * asarray(g) / bc1)
            else:
                m_hat = asarray(m) / bc1
            v_hat = asarray(v) / bc2
            return m_hat / (sqrt(v_hat) + eps)

        out_updates = tree_map(_compute_update, new_mu, new_nu, updates)
        return out_updates, ScaleByBeliefState(count=count_inc, mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: DTypeLike = None,
    *,
    mode: Literal["hard", "smooth", "refined"] = "hard",
    smooth_beta: float = 1.0,
) -> GradientTransformation[ScaleByLionState]:
    """Rescale updates according to the Lion (EvoLved Sign Momentum) algorithm.

    Args:
        b1: Rate for combining current gradient and momentum before taking `sign`.
        b2: Exponential decay rate for updating the momentum trace.
        mu_dtype: Optional data type for the momentum accumulator.
        mode: Sign variant to use: `'hard'` (`sign(c)`), `'smooth'` (`tanh(smooth_beta * c)`),
            or `'refined'` (`clip(c * smooth_beta, -1, 1)`).
        smooth_beta: Smoothing factor used when `mode == 'smooth'` or `'refined'`.

    Returns:
        A `GradientTransformation` implementing Lion sign updates.
    """

    def init_fn(params: Params) -> ScaleByLionState:
        """Initialize `ScaleByLionState` with zero momentum.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByLionState`.
        """
        return ScaleByLionState(
            count=asarray(0),
            mu=tree_zeros_like(params, dtype=mu_dtype),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByLionState]:
        """Compute Lion update (`hard`, `smooth`, or `refined`) and update momentum `b2 * m + (1 - b2) * g`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByLionState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(sign_updates, new_lion_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByLionState`.
        """
        del params
        if not isinstance(state, ScaleByLionState):
            raise TypeError(
                f"Expected ScaleByLionState in scale_by_lion, got {type(state)!r}."
            )
        count_inc = state.count + 1

        def _sign_step(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Compute Lion step from `c = b1 * m + (1 - b1) * g`.

            Args:
                m: Previous momentum leaf.
                g: Current gradient leaf.

            Returns:
                Transformed Lion update Tensor leaf.
            """
            if m is None or g is None:
                return g
            c = b1 * asarray(m) + (1.0 - b1) * asarray(g)
            if mode == "smooth":
                return 2.0 * sigmoid(2.0 * smooth_beta * c) - 1.0
            if mode == "refined":
                return minimum(maximum(smooth_beta * c, -1.0), 1.0)
            return sign(c)

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update momentum `b2 * m + (1 - b2) * g`.

            Args:
                m: Previous momentum leaf.
                g: Current gradient leaf.

            Returns:
                Updated momentum leaf.
            """
            if m is None or g is None:
                return m
            res = b2 * asarray(m) + (1.0 - b2) * asarray(g)
            return asarray(res, dtype=mu_dtype) if mu_dtype is not None else res

        out_updates = tree_map(_sign_step, state.mu, updates)
        new_mu = tree_map(_update_mu, state.mu, updates)
        return out_updates, ScaleByLionState(count=count_inc, mu=new_mu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> GradientTransformation[ScaleByRmsState | ScaleByRmsWithCountState]:
    """Rescale updates by the root of the exponentially moving average of squared gradients.

    Args:
        decay: Exponential decay factor for squared gradients.
        eps: Small numerical stability constant.
        initial_scale: Initial value for the second moment accumulator.
        eps_in_sqrt: Whether to place `eps` inside `sqrt(nu + eps)` or outside `sqrt(nu) + eps`.
        bias_correction: Whether to apply `(1 - decay^t)` bias correction to `nu`.

    Returns:
        A `GradientTransformation` implementing uncentered RMSProp rescaling.
    """

    def init_fn(params: Params) -> ScaleByRmsState | ScaleByRmsWithCountState:
        """Initialize RMS state.

        Args:
            params: Model parameters PyTree.

        Returns:
            `ScaleByRmsWithCountState` if `bias_correction=True`, else `ScaleByRmsState`.
        """

        def _init_nu(p: TensorLeaf) -> TensorLeaf:
            """Fill leaf accumulator with `initial_scale`.

            Args:
                p: Parameter leaf.

            Returns:
                Initialized accumulator leaf.
            """
            return None if p is None else full_like(p, initial_scale)

        nu = tree_map(_init_nu, params)
        if bias_correction:
            return ScaleByRmsWithCountState(count=asarray(0), nu=nu)
        return ScaleByRmsState(nu=nu)

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByRmsState | ScaleByRmsWithCountState]:
        """Update squared gradient EMA and rescale updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current RMS state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(rescaled_updates, new_rms_state)`.

        Raises:
            TypeError: If `state` is not a valid RMS state.
        """
        del params
        if not isinstance(state, (ScaleByRmsState, ScaleByRmsWithCountState)):
            raise TypeError(
                f"Expected ScaleByRmsState or ScaleByRmsWithCountState in scale_by_rms, got {type(state)!r}."
            )
        if isinstance(state, ScaleByRmsWithCountState):
            prev_nu = state.nu
            count_inc = state.count + 1
        else:
            prev_nu = state.nu
            count_inc = asarray(1)

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update EMA `decay * v + (1 - decay) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            return decay * asarray(v) + (1.0 - decay) * (g_t * g_t)

        new_nu = tree_map(_update_nu, prev_nu, updates)
        bc = 1.0 - (decay**count_inc)

        def _scale_leaf(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Rescale gradient leaf `g` using `v`.

            Args:
                v: Updated second moment leaf.
                g: Current gradient leaf.

            Returns:
                Rescaled update leaf.
            """
            if v is None or g is None:
                return g
            v_hat = (asarray(v) / bc) if bias_correction else asarray(v)
            denom = sqrt(v_hat + eps) if eps_in_sqrt else (sqrt(v_hat) + eps)
            return asarray(g) / denom

        out_updates = tree_map(_scale_leaf, new_nu, updates)
        if bias_correction:
            return out_updates, ScaleByRmsWithCountState(count=count_inc, nu=new_nu)
        return out_updates, ScaleByRmsState(nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_stddev(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> GradientTransformation[ScaleByRStdDevState | ScaleByRStdDevWithCountState]:
    """Rescale updates by the root of the centered exponentially moving variance of gradients.

    Args:
        decay: Exponential decay factor for first and second moments.
        eps: Small numerical stability constant.
        initial_scale: Initial value for the second moment accumulator.
        eps_in_sqrt: Whether `eps` is added inside or outside the square root.
        bias_correction: Whether to apply bias correction to both moments.

    Returns:
        A `GradientTransformation` implementing centered RMSProp rescaling.
    """

    def init_fn(params: Params) -> ScaleByRStdDevState | ScaleByRStdDevWithCountState:
        """Initialize centered RMS state.

        Args:
            params: Model parameters PyTree.

        Returns:
            `ScaleByRStdDevWithCountState` if `bias_correction=True`, else `ScaleByRStdDevState`.
        """
        mu = tree_zeros_like(params)

        def _init_nu(p: TensorLeaf) -> TensorLeaf:
            """Fill leaf accumulator with `initial_scale`.

            Args:
                p: Parameter leaf.

            Returns:
                Initialized accumulator leaf.
            """
            return None if p is None else full_like(p, initial_scale)

        nu = tree_map(_init_nu, params)
        if bias_correction:
            return ScaleByRStdDevWithCountState(count=asarray(0), mu=mu, nu=nu)
        return ScaleByRStdDevState(mu=mu, nu=nu)

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByRStdDevState | ScaleByRStdDevWithCountState]:
        """Update centered variance estimate and rescale updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current centered RMS state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(rescaled_updates, new_stddev_state)`.

        Raises:
            TypeError: If `state` is not a valid centered RMS state.
        """
        del params
        if not isinstance(state, (ScaleByRStdDevState, ScaleByRStdDevWithCountState)):
            raise TypeError(
                f"Expected ScaleByRStdDevState or ScaleByRStdDevWithCountState in scale_by_stddev, got {type(state)!r}."
            )
        if isinstance(state, ScaleByRStdDevWithCountState):
            prev_mu = state.mu
            prev_nu = state.nu
            count_inc = state.count + 1
        else:
            prev_mu = state.mu
            prev_nu = state.nu
            count_inc = asarray(1)

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update mean `decay * m + (1 - decay) * g`.

            Args:
                m: Previous mean leaf.
                g: Current gradient leaf.

            Returns:
                Updated mean leaf.
            """
            if m is None or g is None:
                return m
            return decay * asarray(m) + (1.0 - decay) * asarray(g)

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update second moment `decay * v + (1 - decay) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            return decay * asarray(v) + (1.0 - decay) * (g_t * g_t)

        new_mu = tree_map(_update_mu, prev_mu, updates)
        new_nu = tree_map(_update_nu, prev_nu, updates)
        bc = 1.0 - (decay**count_inc)

        def _scale_leaf(m: TensorLeaf, v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Rescale `g` by centered standard deviation `sqrt((v - m^2)_hat)`.

            Args:
                m: Updated mean leaf.
                v: Updated second moment leaf.
                g: Current gradient leaf.

            Returns:
                Rescaled update leaf.
            """
            if m is None or v is None or g is None:
                return g
            m_t = asarray(m)
            v_t = asarray(v)
            m_hat = (m_t / bc) if bias_correction else m_t
            v_hat = (v_t / bc) if bias_correction else v_t
            var = v_hat - (m_hat * m_hat)
            denom = sqrt(var + eps) if eps_in_sqrt else (sqrt(var) + eps)
            return asarray(g) / denom

        out_updates = tree_map(_scale_leaf, new_mu, new_nu, updates)
        if bias_correction:
            return out_updates, ScaleByRStdDevWithCountState(
                count=count_inc, mu=new_mu, nu=new_nu
            )
        return out_updates, ScaleByRStdDevState(mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_yogi(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6,
) -> GradientTransformation[ScaleByAdamState]:
    """Rescale updates according to the Yogi additive adaptive variance algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Adaptation rate for the additive second moment update.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        initial_accumulator_value: Initial value for both first and second moment accumulators.

    Returns:
        A `GradientTransformation` implementing Yogi rescaling.
    """

    def init_fn(params: Params) -> ScaleByAdamState:
        """Initialize `ScaleByAdamState` with `initial_accumulator_value`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByAdamState`.
        """

        def _init_leaf(p: TensorLeaf) -> TensorLeaf:
            """Fill leaf accumulator with `initial_accumulator_value`.

            Args:
                p: Parameter leaf.

            Returns:
                Initialized accumulator leaf.
            """
            return None if p is None else full_like(p, initial_accumulator_value)

        return ScaleByAdamState(
            count=asarray(0),
            mu=tree_map(_init_leaf, params),
            nu=tree_map(_init_leaf, params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByAdamState]:
        """Update Yogi moments and compute bias-corrected rescaled updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByAdamState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(yogi_updates, new_yogi_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByAdamState`.
        """
        del params
        if not isinstance(state, ScaleByAdamState):
            raise TypeError(
                f"Expected ScaleByAdamState in scale_by_yogi, got {type(state)!r}."
            )
        count_inc = state.count + 1

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update first moment `b1 * m + (1 - b1) * g`.

            Args:
                m: Previous first moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated first moment leaf.
            """
            if m is None or g is None:
                return m
            return b1 * asarray(m) + (1.0 - b1) * asarray(g)

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update second moment `v - (1 - b2) * sign(v - g^2) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_sq = asarray(g) * asarray(g)
            v_t = asarray(v)
            return v_t - (1.0 - b2) * sign(v_t - g_sq) * g_sq

        new_mu = tree_map(_update_mu, state.mu, updates)
        new_nu = tree_map(_update_nu, state.nu, updates)
        bc1 = 1.0 - (b1**count_inc)
        bc2 = 1.0 - (b2**count_inc)

        def _scale_yogi(m: TensorLeaf, v: TensorLeaf) -> TensorLeaf:
            """Compute bias-corrected Yogi update leaf.

            Args:
                m: Updated first moment leaf.
                v: Updated second moment leaf.

            Returns:
                Rescaled Yogi update leaf.
            """
            if m is None or v is None:
                return m
            m_hat = asarray(m) / bc1
            v_hat = asarray(v) / bc2
            return m_hat / (sqrt(v_hat + eps_root) + eps)

        out_updates = tree_map(_scale_yogi, new_mu, new_nu)
        return out_updates, ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_novograd(
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: ScalarOrSchedule = 0.0,
    mu_dtype: DTypeLike = None,
) -> GradientTransformation[ScaleByNovogradState]:
    """Rescale updates using layer-wise adaptive second moments according to NovoGrad.

    Args:
        b1: Momentum decay rate for the first moment.
        b2: Exponential decay rate for the per-leaf squared gradient L2 norms.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        weight_decay: Scalar or schedule weight decay rate added to normalized gradients.
        mu_dtype: Optional data type for storing the first moment accumulator.

    Returns:
        A `GradientTransformation` implementing NovoGrad layer-wise rescaling.
    """

    def init_fn(params: Params) -> ScaleByNovogradState:
        """Initialize `ScaleByNovogradState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByNovogradState`.
        """

        def _zero_scalar(p: TensorLeaf) -> TensorLeaf:
            """Initialize per-leaf scalar second moment to `0.0`.

            Args:
                p: Parameter leaf.

            Returns:
                Scalar `0.0` Tensor.
            """
            return None if p is None else asarray(0.0)

        return ScaleByNovogradState(
            count=asarray(0),
            mu=tree_zeros_like(params, dtype=mu_dtype),
            nu=tree_map(_zero_scalar, params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByNovogradState]:
        """Update layer-wise gradient norm EMA and momentum.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByNovogradState`.
            params: Current model parameters PyTree (required when `weight_decay != 0` or matching Optax).

        Returns:
            Tuple of `(novograd_updates, new_novograd_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByNovogradState`.
            ValueError: If `params` is None when required.
        """
        if not isinstance(state, ScaleByNovogradState):
            raise TypeError(
                f"Expected ScaleByNovogradState in scale_by_novograd, got {type(state)!r}."
            )
        wd_val = evaluate_schedule(weight_decay, state.count)
        if params is None:
            if callable(weight_decay) or weight_decay != 0.0:
                raise ValueError(
                    "`params` must be provided to `scale_by_novograd` when weight_decay != 0."
                )
            params_tree: Params = tree_zeros_like(updates)
        else:
            params_tree = params

        count_inc = state.count + 1
        is_first_step = count_inc == 1

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Compute per-leaf squared L2 norm EMA.

            Args:
                v: Previous scalar second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated scalar second moment leaf.
            """
            if v is None or g is None:
                return v
            norm_sq = leaf_l2_norm(g) ** 2
            return where(is_first_step, norm_sq, b2 * asarray(v) + (1.0 - b2) * norm_sq)

        new_nu = tree_map(_update_nu, state.nu, updates)

        def _update_mu(
            m: TensorLeaf, g: TensorLeaf, p: TensorLeaf, v: TensorLeaf
        ) -> TensorLeaf:
            """Update NovoGrad momentum leaf.

            Args:
                m: Previous momentum leaf.
                g: Current gradient leaf.
                p: Current parameter leaf.
                v: Updated scalar second moment leaf.

            Returns:
                Updated momentum leaf.
            """
            if m is None or g is None or p is None or v is None:
                return g
            grad_normed = asarray(g) / (
                sqrt(asarray(v) + eps_root) + eps
            ) + wd_val * asarray(p)
            res = where(is_first_step, grad_normed, b1 * asarray(m) + grad_normed)
            return asarray(res, dtype=mu_dtype) if mu_dtype is not None else res

        new_mu = tree_map(_update_mu, state.mu, updates, params_tree, new_nu)
        return new_mu, ScaleByNovogradState(count=count_inc, mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_trust_ratio(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.0,
    eps: float = 0.0,
) -> GradientTransformation[ScaleByTrustRatioState]:
    """Scale updates by the layer-wise ratio of parameter norm to update norm.

    Args:
        min_norm: Minimum floor clipped onto both parameter and update norms.
        trust_coefficient: Multiplier applied to the trust ratio.
        eps: Additive constant in the trust ratio denominator.

    Returns:
        A `GradientTransformation` scaling each leaf by its trust ratio.
    """

    def init_fn(params: Params) -> ScaleByTrustRatioState:
        """Initialize empty state for `scale_by_trust_ratio`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            A `ScaleByTrustRatioState` instance.
        """
        del params
        return ScaleByTrustRatioState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByTrustRatioState]:
        """Scale each gradient leaf by `trust_coefficient * ||param|| / (||update|| + eps)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(scaled_updates, ScaleByTrustRatioState())`.

        Raises:
            ValueError: If `params` is None.
        """
        del state
        if params is None:
            raise ValueError(
                "`params` must be provided to `scale_by_trust_ratio` update."
            )

        def _scale_leaf(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Scale leaf `u` by trust ratio relative to `p`.

            Args:
                u: Gradient update leaf.
                p: Parameter leaf.

            Returns:
                Scaled update leaf.
            """
            if u is None or p is None:
                return u
            u_t = asarray(u)
            p_t = asarray(p)
            p_norm = safe_norm(p_t, min_norm)
            u_norm = safe_norm(u_t, min_norm)
            trust_ratio = trust_coefficient * p_norm / (u_norm + eps)
            zero_norm = (p_norm == 0.0) | (u_norm == 0.0)
            safe_ratio = where(zero_norm, 1.0, trust_ratio)
            return u_t * safe_ratio

        return tree_map(_scale_leaf, updates, params), ScaleByTrustRatioState()

    return GradientTransformation(init_fn, update_fn)


def add_decayed_weights(
    weight_decay: ScalarOrSchedule = 0.0,
    mask: MaskOrFn = None,
) -> GradientTransformation[AddDecayedWeightsState]:
    """Add parameter weight decay (`weight_decay * params`) to gradient updates.

    Args:
        weight_decay: Scalar or callable schedule specifying the weight decay strength.
        mask: Optional boolean mask PyTree or callable `(Params) -> MaskTree` selecting
            which parameters receive weight decay.

    Returns:
        A `GradientTransformation` adding decayed weights to updates.
    """

    def init_fn(params: Params) -> AddDecayedWeightsState:
        """Initialize `AddDecayedWeightsState` with `count = 0`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `AddDecayedWeightsState`.
        """
        del params
        return AddDecayedWeightsState(count=asarray(0))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AddDecayedWeightsState]:
        """Add `weight_decay * params` to `updates` where unmasked.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AddDecayedWeightsState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(decayed_updates, new_state)`.

        Raises:
            TypeError: If `state` is not an `AddDecayedWeightsState`.
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError(
                "`params` must be provided to `add_decayed_weights` update."
            )
        if not isinstance(state, AddDecayedWeightsState):
            raise TypeError(
                f"Expected AddDecayedWeightsState in add_decayed_weights, got {type(state)!r}."
            )
        count = state.count
        wd = evaluate_schedule(weight_decay, count)

        def _add_wd(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Add `wd * p` to `u`.

            Args:
                u: Update leaf.
                p: Parameter leaf.

            Returns:
                Decayed update leaf.
            """
            if u is None or p is None:
                return u
            return asarray(u) + wd * asarray(p)

        decayed = tree_map(_add_wd, updates, params)
        mask_tree = resolve_mask(mask, params)
        if mask_tree is not None:
            decayed = apply_mask_tree(decayed, updates, mask_tree)
        return decayed, AddDecayedWeightsState(count=count + 1)

    return GradientTransformation(init_fn, update_fn)


class ZeroNansState(NamedTuple):
    """State for `zero_nans` recording whether NaNs were found in each parameter leaf.

    Attributes:
        found_nan: PyTree of boolean scalar Tensors indicating whether NaNs occurred.
    """

    found_nan: PyTree


@dataclass(frozen=True)
class AddNoiseState(_OptStateDataclassBase):
    """State for `add_noise` holding the step count and PRNG key.

    Attributes:
        count: Optimization step counter Tensor.
        rng_key: PRNG key Tensor.
    """

    count: Tensor
    rng_key: Tensor
    _fields: tuple[str, ...] = ("count", "rng_key")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'` and `'rng_key'`.
        """
        return {"count": self.count, "rng_key": self.rng_key}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> AddNoiseState:
        """Construct `AddNoiseState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, rng_key)`.

        Returns:
            Reconstructed `AddNoiseState`.
        """
        items = list(iterable)
        return cls(
            count=_as_count_tensor(items[0]),
            rng_key=_as_count_tensor(items[1]),
        )


@dataclass(frozen=True)
class EmaState(_OptStateDataclassBase):
    """State for `ema` holding the step count and exponential moving average PyTree.

    Attributes:
        count: Optimization step counter Tensor.
        ema: Exponential moving average of updates PyTree.
    """

    count: Tensor
    ema: Updates
    _fields: tuple[str, ...] = ("count", "ema")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'` and `'ema'`.
        """
        return {"count": self.count, "ema": self.ema}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> EmaState:
        """Construct `EmaState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, ema)`.

        Returns:
            Reconstructed `EmaState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), ema=items[1])


@dataclass(frozen=True)
class ScaleByAmsgradState(_OptStateDataclassBase):
    """State for `scale_by_amsgrad` holding `count`, `mu`, `nu`, and `nu_max`.

    Attributes:
        count: Optimization step counter Tensor.
        mu: First moment accumulator PyTree.
        nu: Second moment accumulator PyTree.
        nu_max: Running maximum of second moment accumulator PyTree.
    """

    count: Tensor
    mu: Updates
    nu: Updates
    nu_max: Updates
    _fields: tuple[str, ...] = ("count", "mu", "nu", "nu_max")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'`, `'mu'`, `'nu'`, and `'nu_max'`.
        """
        return {
            "count": self.count,
            "mu": self.mu,
            "nu": self.nu,
            "nu_max": self.nu_max,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByAmsgradState:
        """Construct `ScaleByAmsgradState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu, nu_max)`.

        Returns:
            Reconstructed `ScaleByAmsgradState`.
        """
        items = list(iterable)
        return cls(
            count=_as_count_tensor(items[0]),
            mu=items[1],
            nu=items[2],
            nu_max=items[3],
        )


class KeepParamsNonnegativeState(NamedTuple):
    """Empty state for `keep_params_nonnegative`."""


def zero_nans() -> GradientTransformation[ZeroNansState]:
    """Replace `NaN` values in gradient updates with `0.0` and record boolean flags in state.

    Returns:
        A `GradientTransformation` zeroing out `NaN` values.
    """

    def init_fn(params: Params) -> ZeroNansState:
        """Initialize `ZeroNansState` with `False` boolean flags for each leaf.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ZeroNansState`.
        """

        def _false_leaf(p: TensorLeaf) -> TensorLeaf:
            """Return scalar `False` tensor for leaf `p`.

            Args:
                p: Parameter leaf.

            Returns:
                Scalar boolean Tensor `False`.
            """
            return None if p is None else asarray(False)

        return ZeroNansState(found_nan=tree_map(_false_leaf, params))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ZeroNansState]:
        """Replace `NaN` entries in `updates` with `0.0` and record `found_nan`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ZeroNansState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(clean_updates, ZeroNansState(found_nan))`.
        """
        del state, params

        def _check_nan(u: TensorLeaf) -> TensorLeaf:
            """Return scalar boolean Tensor indicating if `u` contains any `NaN`.

            Args:
                u: Gradient update leaf.

            Returns:
                Scalar boolean Tensor.
            """
            if u is None:
                return None
            u_t = asarray(u)
            return reduce_sum(u_t != u_t) > 0

        def _clean_nan(u: TensorLeaf) -> TensorLeaf:
            """Replace any `NaN` elements in `u` with `0.0`.

            Args:
                u: Gradient update leaf.

            Returns:
                Cleaned Tensor leaf.
            """
            if u is None:
                return None
            u_t = asarray(u)
            return asarray(where(u_t != u_t, 0.0, u_t), dtype=u_t.dtype)

        return (
            tree_map(_clean_nan, updates),
            ZeroNansState(found_nan=tree_map(_check_nan, updates)),
        )

    return GradientTransformation(init_fn, update_fn)


def centralize() -> GradientTransformation[EmptyState]:
    """Centralize gradients by subtracting the mean across all axes except axis 0 for rank >= 2.

    Returns:
        A `GradientTransformation` centralizing multi-dimensional gradient leaves.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `centralize`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `EmptyState`.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Subtract mean across axes `1..ndim-1` for leaves with `ndim >= 2`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(centralized_updates, EmptyState())`.
        """
        del state, params

        def _centralize_leaf(g: TensorLeaf) -> TensorLeaf:
            """Centralize single leaf `g` if `g.ndim >= 2`.

            Args:
                g: Gradient update leaf.

            Returns:
                Centralized Tensor leaf.
            """
            if g is None:
                return None
            g_t = asarray(g)
            if g_t.ndim <= 1:
                return g_t
            axes = tuple(range(1, g_t.ndim))
            return g_t - reduce_mean(g_t, axis=axes, keepdims=True)

        return tree_map(_centralize_leaf, updates), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def keep_params_nonnegative() -> GradientTransformation[KeepParamsNonnegativeState]:
    """Clamp updates so that `params + updates >= 0` (`updates' = max(updates, -params)`).

    Returns:
        A `GradientTransformation` keeping parameters non-negative after `apply_updates`.
    """

    def init_fn(params: Params) -> KeepParamsNonnegativeState:
        """Initialize empty `KeepParamsNonnegativeState`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `KeepParamsNonnegativeState`.
        """
        del params
        return KeepParamsNonnegativeState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, KeepParamsNonnegativeState]:
        """Clamp each update leaf below at `-params`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `KeepParamsNonnegativeState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(clamped_updates, KeepParamsNonnegativeState())`.

        Raises:
            ValueError: If `params` is None.
        """
        del state
        if params is None:
            raise ValueError(
                "`params` must be provided to `keep_params_nonnegative` update."
            )

        def _clamp_leaf(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Clamp `u` so `p + u >= 0`.

            Args:
                u: Update leaf.
                p: Parameter leaf.

            Returns:
                Clamped update leaf.
            """
            if u is None or p is None:
                return u
            u_t = asarray(u)
            p_t = asarray(p)
            return where((p_t + u_t) < 0.0, -p_t, u_t)

        return tree_map(_clamp_leaf, updates, params), KeepParamsNonnegativeState()

    return GradientTransformation(init_fn, update_fn)


def add_noise(
    eta: float,
    gamma: float,
    key: RngKeyLike | None = None,
    *,
    seed: RngKeyLike | None = None,
) -> GradientTransformation[AddNoiseState]:
    """Add annealed Gaussian noise `N(0, eta / (1 + t) ** gamma)` to updates.

    Args:
        eta: Base variance of the Gaussian noise.
        gamma: Decay exponent controlling noise annealing over steps `t`.
        key: Optional `zero_jax.random.PRNGKey`, `zero_flax.nnx.Rngs`, or integer seed.
        seed: Optional deprecated seed or PRNG key keyword argument (used if `key is None`).

    Returns:
        A `GradientTransformation` adding annealed Gaussian noise.

    Raises:
        ValueError: If both `key` and `seed` are provided.
    """
    if key is None and seed is None:
        init_seed = 0
    elif key is not None and seed is not None:
        raise ValueError("Provide only one of `key` or `seed` to `add_noise`.")
    elif key is not None:
        init_seed = _extract_seed_int(key)
    else:
        init_seed = _extract_seed_int(seed) if seed is not None else 0

    def init_fn(params: Params) -> AddNoiseState:
        """Initialize `AddNoiseState` with `count = 0` and initial PRNG key.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `AddNoiseState`.
        """
        del params
        return AddNoiseState(
            count=asarray(0),
            rng_key=asarray([0, init_seed], dtype="int32"),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AddNoiseState]:
        """Add annealed Gaussian noise to `updates` and advance `state`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AddNoiseState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(noisy_updates, new_state)`.

        Raises:
            TypeError: If `state` is not an `AddNoiseState`.
        """
        del params
        if not isinstance(state, AddNoiseState):
            raise TypeError(
                f"Expected AddNoiseState in add_noise, got {type(state)!r}."
            )
        count_inc = state.count + 1
        t_f = asarray(count_inc, dtype="float32")
        variance = eta / power(t_f, float(gamma))
        std = sqrt(maximum(variance, 0.0))
        subkeys = _split_seed_to_keys(_extract_seed_int(state.rng_key), 2)
        next_key, sample_key = subkeys[0], subkeys[1]
        noise_tree = tree_random_like(sample_key, updates)

        def _add_noise_leaf(u: TensorLeaf, n: TensorLeaf) -> TensorLeaf:
            """Add `std * n` to leaf `u`.

            Args:
                u: Update leaf.
                n: Standard normal noise leaf.

            Returns:
                Noisy update leaf.
            """
            if u is None or n is None:
                return u
            return asarray(u) + std * asarray(n)

        return (
            tree_map(_add_noise_leaf, updates, noise_tree),
            AddNoiseState(count=count_inc, rng_key=next_key),
        )

    return GradientTransformation(init_fn, update_fn)


def ema(
    decay: float,
    debias: bool = True,
    accumulator_dtype: DTypeLike = None,
) -> GradientTransformation[EmaState]:
    """Compute an exponential moving average of past updates.

    Args:
        decay: Exponential decay rate in `[0, 1)`.
        debias: Whether to apply zero-debiasing `1 / (1 - decay ** t)`.
        accumulator_dtype: Optional data type for the EMA state accumulator.

    Returns:
        A `GradientTransformation` computing the EMA of updates.
    """

    def init_fn(params: Params) -> EmaState:
        """Initialize `EmaState` with zero EMA accumulators.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `EmaState`.
        """
        return EmaState(
            count=asarray(0),
            ema=tree_zeros_like(params, dtype=accumulator_dtype),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmaState]:
        """Update EMA state and return (optionally debiased) EMA updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `EmaState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(ema_updates, new_ema_state)`.

        Raises:
            TypeError: If `state` is not an `EmaState`.
        """
        del params
        if not isinstance(state, EmaState):
            raise TypeError(f"Expected EmaState in ema, got {type(state)!r}.")
        count_inc = state.count + 1

        def _ema_step(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update EMA leaf `decay * m + (1 - decay) * g`.

            Args:
                m: Previous EMA leaf.
                g: Current update leaf.

            Returns:
                Updated EMA leaf.
            """
            if m is None or g is None:
                return g
            val = decay * asarray(m) + (1.0 - decay) * asarray(g)
            return (
                asarray(val, dtype=accumulator_dtype)
                if accumulator_dtype is not None
                else val
            )

        new_ema = tree_map(_ema_step, state.ema, updates)
        if not debias:
            return new_ema, EmaState(count=count_inc, ema=new_ema)
        denom = 1.0 - power(asarray(decay, dtype="float32"), count_inc)
        safe_denom = where(abs_tensor(denom) > 1e-12, denom, 1.0)

        def _debias_leaf(m: TensorLeaf) -> TensorLeaf:
            """Divide EMA leaf by `1 - decay ** t`.

            Args:
                m: EMA leaf.

            Returns:
                Debiased EMA leaf.
            """
            return None if m is None else asarray(m) / safe_denom

        return (
            tree_map(_debias_leaf, new_ema),
            EmaState(count=count_inc, ema=new_ema),
        )

    return GradientTransformation(init_fn, update_fn)


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: DTypeLike = None,
    bias_correction_mu: bool = True,
    bias_correction_nu: bool = True,
) -> GradientTransformation[ScaleByAmsgradState]:
    """Rescale updates according to the AMSGrad algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added to the denominator outside the square root.
        eps_root: Small constant added inside the square root.
        mu_dtype: Optional dtype for the first moment accumulator `mu`.
        bias_correction_mu: Whether to apply bias correction to `mu`.
        bias_correction_nu: Whether to apply bias correction to `nu` before `nu_max`.

    Returns:
        A `GradientTransformation` implementing AMSGrad scaling.
    """

    def init_fn(params: Params) -> ScaleByAmsgradState:
        """Initialize `ScaleByAmsgradState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByAmsgradState`.
        """
        return ScaleByAmsgradState(
            count=asarray(0),
            mu=tree_zeros_like(params, dtype=mu_dtype),
            nu=tree_zeros_like(params),
            nu_max=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByAmsgradState]:
        """Update `mu`, `nu`, and `nu_max` and scale updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByAmsgradState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(scaled_updates, new_amsgrad_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByAmsgradState`.
        """
        del params
        if not isinstance(state, ScaleByAmsgradState):
            raise TypeError(
                f"Expected ScaleByAmsgradState in scale_by_amsgrad, got {type(state)!r}."
            )
        count_inc = state.count + 1
        bc1 = (
            1.0 - power(asarray(b1, dtype="float32"), count_inc)
            if bias_correction_mu
            else asarray(1.0)
        )
        bc2 = (
            1.0 - power(asarray(b2, dtype="float32"), count_inc)
            if bias_correction_nu
            else asarray(1.0)
        )
        safe_bc1 = where(abs_tensor(bc1) > 1e-12, bc1, 1.0)
        safe_bc2 = where(abs_tensor(bc2) > 1e-12, bc2, 1.0)

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update first moment `b1 * m + (1 - b1) * g`.

            Args:
                m: Previous first moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated first moment leaf.
            """
            if m is None or g is None:
                return m
            val = b1 * asarray(m) + (1.0 - b1) * asarray(g)
            return asarray(val, dtype=mu_dtype) if mu_dtype is not None else val

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update second moment `b2 * v + (1 - b2) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            return b2 * asarray(v) + (1.0 - b2) * (g_t * g_t)

        def _update_nu_max(v_max: TensorLeaf, v_new: TensorLeaf) -> TensorLeaf:
            """Update running maximum `max(v_max, v_new / bc2)`.

            Args:
                v_max: Previous running maximum leaf.
                v_new: Updated second moment leaf.

            Returns:
                Updated `nu_max` leaf.
            """
            if v_max is None or v_new is None:
                return v_max
            v_hat = asarray(v_new) / safe_bc2
            return maximum(asarray(v_max), v_hat)

        def _compute_step(m: TensorLeaf, v_max: TensorLeaf) -> TensorLeaf:
            """Compute `(m / bc1) / (sqrt(v_max + eps_root) + eps)`.

            Args:
                m: Updated first moment leaf.
                v_max: Updated running max second moment leaf.

            Returns:
                Scaled update leaf.
            """
            if m is None or v_max is None:
                return m
            m_hat = asarray(m) / safe_bc1
            return m_hat / (sqrt(asarray(v_max) + eps_root) + eps)

        new_mu = tree_map(_update_mu, state.mu, updates)
        new_nu = tree_map(_update_nu, state.nu, updates)
        new_nu_max = tree_map(_update_nu_max, state.nu_max, new_nu)
        out_updates = tree_map(_compute_step, new_mu, new_nu_max)
        return (
            out_updates,
            ScaleByAmsgradState(
                count=count_inc,
                mu=new_mu,
                nu=new_nu,
                nu_max=new_nu_max,
            ),
        )

    return GradientTransformation(init_fn, update_fn)


def scale_by_param_block_norm(
    min_scale: float = 0.001,
) -> GradientTransformation[EmptyState]:
    """Scale updates for each parameter block by `max(||param||_2, min_scale)`.

    Args:
        min_scale: Minimum scaling factor applied when parameter norm is small.

    Returns:
        A `GradientTransformation` scaling updates by parameter block L2 norm.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `scale_by_param_block_norm`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `EmptyState`.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Scale each update leaf by `max(leaf_l2_norm(p), min_scale)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(scaled_updates, EmptyState())`.

        Raises:
            ValueError: If `params` is None.
        """
        del state
        if params is None:
            raise ValueError(
                "`params` must be provided to `scale_by_param_block_norm` update."
            )

        def _scale_block(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Multiply `u` by `max(||p||_2, min_scale)`.

            Args:
                u: Update leaf.
                p: Parameter leaf.

            Returns:
                Scaled update leaf.
            """
            if u is None or p is None:
                return u
            return asarray(u) * maximum(leaf_l2_norm(p), min_scale)

        return tree_map(_scale_block, updates, params), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def scale_by_param_block_rms(
    min_scale: float = 0.001,
) -> GradientTransformation[EmptyState]:
    """Scale updates for each parameter block by `max(rms(param), min_scale)`.

    Args:
        min_scale: Minimum scaling factor applied when parameter RMS is small.

    Returns:
        A `GradientTransformation` scaling updates by parameter block RMS.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `scale_by_param_block_rms`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `EmptyState`.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Scale each update leaf by `max(rms(p), min_scale)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(scaled_updates, EmptyState())`.

        Raises:
            ValueError: If `params` is None.
        """
        del state
        if params is None:
            raise ValueError(
                "`params` must be provided to `scale_by_param_block_rms` update."
            )

        def _scale_rms(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Multiply `u` by `max(rms(p), min_scale)`.

            Args:
                u: Update leaf.
                p: Parameter leaf.

            Returns:
                Scaled update leaf.
            """
            if u is None or p is None:
                return u
            p_t = asarray(p)
            rms = sqrt(reduce_mean(p_t * p_t))
            return asarray(u) * maximum(rms, min_scale)

        return tree_map(_scale_rms, updates, params), EmptyState()

    return GradientTransformation(init_fn, update_fn)


ScaleByAddNoiseState = AddNoiseState


__all__ = [
    "AddDecayedWeightsState",
    "AddNoiseState",
    "EmaState",
    "KeepParamsNonnegativeState",
    "ScaleByAdamState",
    "ScaleByAdanState",
    "ScaleByAddNoiseState",
    "ScaleByAmsgradState",
    "ScaleByBeliefState",
    "ScaleByLionState",
    "ScaleByNovogradState",
    "ScaleByRStdDevState",
    "ScaleByRStdDevWithCountState",
    "ScaleByRmsState",
    "ScaleByRmsWithCountState",
    "ScaleByRssState",
    "ScaleByTrustRatioState",
    "ZeroNansState",
    "add_decayed_weights",
    "add_noise",
    "centralize",
    "ema",
    "keep_params_nonnegative",
    "scale",
    "scale_by_adam",
    "scale_by_amsgrad",
    "scale_by_belief",
    "scale_by_learning_rate",
    "scale_by_lion",
    "scale_by_novograd",
    "scale_by_param_block_norm",
    "scale_by_param_block_rms",
    "scale_by_rms",
    "scale_by_rss",
    "scale_by_schedule",
    "scale_by_stddev",
    "scale_by_trust_ratio",
    "scale_by_yogi",
    "trace",
    "zero_nans",
]
