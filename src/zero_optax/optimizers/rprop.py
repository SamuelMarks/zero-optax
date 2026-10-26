"""Resilient backpropagation (Rprop) and SM3 optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from zero_optax._tree import (
    _coerce_pytree_child,
    _extract_leaf,
    _rebuild_mapping,
    _rebuild_sequence,
    _unwrap_leaf,
    asarray,
    full_like,
    maximum,
    minimum,
    reduce_max,
    reshape,
    sign,
    sqrt,
    tree_map,
    tree_zeros_like,
    where,
    zeros,
)
from zero_optax._types import (
    CustomMappingPyTreeProtocol,
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
    _OptStateDataclassBase,
)
from zero_optax.combine import chain
from zero_optax.transform import scale, scale_by_learning_rate


@dataclass(frozen=True)
class ScaleByRpropState(_OptStateDataclassBase):
    """State for `scale_by_rprop` holding adaptive step sizes and previous updates.

    Attributes:
        step_sizes: Per-parameter adaptive step sizes PyTree.
        prev_updates: Previous step updates PyTree.
    """

    step_sizes: Updates
    prev_updates: Updates
    _fields: tuple[str, ...] = ("step_sizes", "prev_updates")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'step_sizes'` and `'prev_updates'`.
        """
        return {
            "step_sizes": self.step_sizes,
            "prev_updates": self.prev_updates,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByRpropState:
        """Construct `ScaleByRpropState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(step_sizes, prev_updates)`.

        Returns:
            Reconstructed `ScaleByRpropState`.
        """
        items = list(iterable)
        return cls(step_sizes=items[0], prev_updates=items[1])


def scale_by_rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> GradientTransformation[ScaleByRpropState]:
    """Scale updates with the Rprop (resilient backpropagation) rule.

    Args:
        learning_rate: Initial step size for all parameters.
        eta_minus: Multiplicative decrease factor (`< 1`) when gradient changes sign.
        eta_plus: Multiplicative increase factor (`> 1`) when gradient keeps the same sign.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.

    Returns:
        A `GradientTransformation` implementing Rprop step-size adaptation.
    """

    def init_fn(params: Params) -> ScaleByRpropState:
        """Initialize `ScaleByRpropState` with `learning_rate` step sizes and zero `prev_updates`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByRpropState`.
        """

        def _init_step(p: TensorLeaf) -> TensorLeaf:
            """Fill parameter leaf shape with `learning_rate`.

            Args:
                p: Parameter leaf.

            Returns:
                Constant `learning_rate` Tensor leaf.
            """
            return None if p is None else full_like(p, float(learning_rate))

        return ScaleByRpropState(
            step_sizes=tree_map(_init_step, params),
            prev_updates=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByRpropState]:
        """Adapt per-parameter step sizes based on sign agreement and return Rprop updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByRpropState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(rprop_updates, new_rprop_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByRpropState`.
        """
        del params
        if not isinstance(state, ScaleByRpropState):
            raise TypeError(
                f"Expected ScaleByRpropState in scale_by_rprop, got {type(state)!r}."
            )

        def _adapt_step(
            prev_step: TensorLeaf,
            g: TensorLeaf,
            prev_u: TensorLeaf,
        ) -> TensorLeaf:
            """Update step size leaf based on `g * prev_u` sign.

            Args:
                prev_step: Previous step size leaf.
                g: Current gradient leaf.
                prev_u: Previous update leaf.

            Returns:
                Clipped updated step size leaf.
            """
            if prev_step is None or g is None or prev_u is None:
                return prev_step
            prod = asarray(g) * asarray(prev_u)
            st_t = asarray(prev_step)
            scaled = st_t * where(prod > 0.0, eta_plus, eta_minus)
            clipped = minimum(maximum(scaled, min_step_size), max_step_size)
            return where(prod == 0.0, st_t, clipped)

        def _next_prev_u(
            g: TensorLeaf,
            prev_u: TensorLeaf,
            step_sz: TensorLeaf,
        ) -> TensorLeaf:
            """Compute new `prev_updates` leaf (`0` where `g * prev_u < 0`, else `step_size * sign(g)`).

            Args:
                g: Current gradient leaf.
                prev_u: Previous update leaf.
                step_sz: Updated step size leaf.

            Returns:
                New `prev_updates` leaf.
            """
            if g is None or prev_u is None or step_sz is None:
                return g
            g_t = asarray(g)
            prod = g_t * asarray(prev_u)
            return where(prod < 0.0, 0.0, asarray(step_sz) * sign(g_t))

        def _compute_upd(
            g: TensorLeaf,
            prev_u: TensorLeaf,
        ) -> TensorLeaf:
            """Compute emitted update (`0` where `g * prev_u < 0`, else `prev_u`).

            Args:
                g: Current gradient leaf.
                prev_u: Previous update leaf (`state.prev_updates`).

            Returns:
                Emitted Rprop update leaf.
            """
            if g is None or prev_u is None:
                return prev_u
            prod = asarray(g) * asarray(prev_u)
            return where(prod < 0.0, 0.0, asarray(prev_u))

        new_step_sizes = tree_map(
            _adapt_step, state.step_sizes, updates, state.prev_updates
        )
        new_prev_updates = tree_map(
            _next_prev_u, updates, state.prev_updates, new_step_sizes
        )
        out_updates = tree_map(_compute_upd, updates, state.prev_updates)
        return (
            out_updates,
            ScaleByRpropState(
                step_sizes=new_step_sizes,
                prev_updates=new_prev_updates,
            ),
        )

    return GradientTransformation(init_fn, update_fn)


def rprop(
    learning_rate: ScalarOrSchedule,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The Rprop (resilient backpropagation) optimizer.

    Args:
        learning_rate: Initial step size or learning rate `Schedule`.
        eta_minus: Multiplicative factor (`< 1`) for decreasing step size on sign change.
        eta_plus: Multiplicative factor (`> 1`) for increasing step size on sign agreement.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.

    Returns:
        A `GradientTransformation` implementing Rprop.
    """
    if callable(learning_rate):
        return chain(
            scale_by_rprop(
                1.0,
                eta_minus=eta_minus,
                eta_plus=eta_plus,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
            ),
            scale_by_learning_rate(learning_rate),
        )
    return chain(
        scale_by_rprop(
            float(learning_rate),
            eta_minus=eta_minus,
            eta_plus=eta_plus,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
        ),
        scale(-1.0),
    )


class ScaleBySm3State(NamedTuple):
    """State for `scale_by_sm3` holding momentum `mu` and accumulator `nu`.

    Attributes:
        mu: Momentum accumulator PyTree.
        nu: Second-moment cover accumulator PyTree (each parameter leaf holds a list of 1D rank covers).
    """

    mu: Updates
    nu: Updates


def _init_sm3_nu_tree(tree: Params) -> Updates:
    """Initialize per-parameter-leaf list of 1D rank-cover accumulators `[zeros((d_i,))]`.

    Args:
        tree: Model parameters PyTree.

    Returns:
        PyTree matching `tree` where each tensor leaf is replaced by `list[Tensor]`.
    """
    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        return _rebuild_mapping(
            tree,
            {k: _init_sm3_nu_tree(_coerce_pytree_child(tree[k])) for k in tree},
        )
    if isinstance(tree, (tuple, list)):
        return _rebuild_sequence(tree, [_init_sm3_nu_tree(item) for item in tree])
    unwrapped = _unwrap_leaf(_extract_leaf(tree))
    if unwrapped is None:
        return None
    p_t = asarray(unwrapped)
    if p_t.ndim == 0:
        return [zeros((), dtype=p_t.dtype)]
    return [zeros((dim_sz,), dtype=p_t.dtype) for dim_sz in p_t.shape]


def _update_sm3_leaf(
    g_leaf: TensorLeaf,
    m_leaf: TensorLeaf,
    nu_node: PyTree,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[TensorLeaf, PyTree]:
    """Update a single parameter leaf's SM3 rank covers and momentum step.

    Args:
        g_leaf: Gradient leaf value.
        m_leaf: Previous momentum leaf value.
        nu_node: Previous list of 1D rank-cover Tensors.
        b1: Momentum decay rate in `[0, 1)`.
        b2: Second-moment accumulation weight.
        eps: Numerical stability epsilon.

    Returns:
        Tuple `(new_mu_leaf, new_nu_covers)`.
    """
    u_g = _unwrap_leaf(g_leaf)
    u_m = _unwrap_leaf(m_leaf)
    if u_g is None or u_m is None or not isinstance(nu_node, list):
        return g_leaf, nu_node
    g_t = asarray(u_g)
    m_t = asarray(u_m)
    shape = g_t.shape
    ndim = g_t.ndim
    covers: list[Tensor] = [
        c if isinstance(c, Tensor) else zeros((), dtype=g_t.dtype) for c in nu_node
    ]

    if ndim == 0:
        v_min = covers[0]
    else:
        bcast_first = [1] * ndim
        bcast_first[0] = shape[0]
        v_min = reshape(covers[0], bcast_first)
        for ax in range(1, ndim):
            bcast_ax = [1] * ndim
            bcast_ax[ax] = shape[ax]
            v_min = minimum(v_min, reshape(covers[ax], bcast_ax))

    weight = 1.0 if b2 == 1.0 else (1.0 - b2)
    v_hat = b2 * v_min + weight * (g_t * g_t)

    if ndim <= 1:
        new_covers: PyTree = [v_hat]
    else:
        new_covers = [
            reduce_max(
                v_hat, axis=tuple(i for i in range(ndim) if i != ax), keepdims=False
            )
            for ax in range(ndim)
        ]

    scaled = where(v_hat > 0.0, g_t / sqrt(v_hat + eps), 0.0)
    new_m: TensorLeaf = (b1 * m_t + (1.0 - b1) * scaled) if b1 > 0.0 else scaled
    return new_m, new_covers


def _update_sm3_trees(
    updates: Updates,
    mu: Updates,
    nu: Updates,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[Updates, Updates]:
    """Co-traverse `updates`, `mu`, and `nu` to compute SM3 updates and rank-cover state.

    Args:
        updates: Gradient updates PyTree.
        mu: Previous momentum PyTree.
        nu: Previous SM3 rank-cover PyTree.
        b1: Momentum decay rate.
        b2: Second-moment decay rate.
        eps: Numerical stability constant.

    Returns:
        Tuple `(new_mu, new_nu)`.
    """
    if (
        isinstance(updates, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(mu, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(nu, (Mapping, CustomMappingPyTreeProtocol))
    ):
        mu_map: dict[str, PyTree] = {}
        nu_map: dict[str, PyTree] = {}
        for k in updates:
            mu_map[k], nu_map[k] = _update_sm3_trees(
                _coerce_pytree_child(updates[k]),
                _coerce_pytree_child(mu[k]),
                _coerce_pytree_child(nu[k]),
                b1,
                b2,
                eps,
            )
        return _rebuild_mapping(updates, mu_map), _rebuild_mapping(nu, nu_map)
    if (
        isinstance(updates, (tuple, list))
        and isinstance(mu, (tuple, list))
        and isinstance(nu, (tuple, list))
    ):
        mu_items: list[PyTree] = []
        nu_items: list[PyTree] = []
        for u_item, m_item, n_item in zip(updates, mu, nu):
            m_out, n_out = _update_sm3_trees(u_item, m_item, n_item, b1, b2, eps)
            mu_items.append(m_out)
            nu_items.append(n_out)
        return _rebuild_sequence(updates, mu_items), _rebuild_sequence(nu, nu_items)
    return _update_sm3_leaf(_extract_leaf(updates), _extract_leaf(mu), nu, b1, b2, eps)


def scale_by_sm3(
    b1: float = 0.9,
    b2: float = 1.0,
    eps: float = 1e-8,
) -> GradientTransformation[ScaleBySm3State]:
    """Scale updates by `sm3` (Square-root of Minima of Sums of Maxima of Squared-gradients).

    Args:
        b1: Momentum decay rate in `[0, 1)`.
        b2: Second-moment accumulation weight (1.0 for standard SM3).
        eps: Small constant added to denominator to prevent division by zero.

    Returns:
        A `GradientTransformation` implementing SM3 scaling.
    """

    def init_fn(params: Params) -> ScaleBySm3State:
        """Initialize `ScaleBySm3State` with zero `mu` and rank-1 cover lists `nu`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleBySm3State`.
        """
        return ScaleBySm3State(
            mu=tree_zeros_like(params),
            nu=_init_sm3_nu_tree(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleBySm3State]:
        """Update SM3 rank-cover accumulators and scale updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleBySm3State`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(sm3_updates, new_sm3_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleBySm3State`.
        """
        del params
        if not isinstance(state, ScaleBySm3State):
            raise TypeError(
                f"Expected ScaleBySm3State in scale_by_sm3, got {type(state)!r}."
            )
        new_mu, new_nu = _update_sm3_trees(
            updates, state.mu, state.nu, b1=b1, b2=b2, eps=eps
        )
        return new_mu, ScaleBySm3State(mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def sm3(
    learning_rate: ScalarOrSchedule,
    momentum: float = 0.9,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients) optimizer.

    Args:
        learning_rate: Global scaling factor (`step_size`) or learning rate `Schedule`.
        momentum: Momentum decay rate (default `0.9`).

    Returns:
        A `GradientTransformation` implementing SM3.
    """
    return chain(
        scale_by_sm3(b1=momentum),
        scale_by_learning_rate(learning_rate),
    )


__all__ = [
    "ScaleByRpropState",
    "ScaleBySm3State",
    "rprop",
    "scale_by_rprop",
    "scale_by_sm3",
    "sm3",
]
