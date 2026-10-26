"""AdaMax optimizer (infinity-norm Adam variant)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zero_optax._tree import (
    abs_tensor,
    asarray,
    maximum,
    power,
    tree_map,
    tree_zeros_like,
    where,
)
from zero_optax._types import (
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
)
from zero_optax.combine import chain
from zero_optax.transform import scale_by_learning_rate


@dataclass(frozen=True)
class ScaleByAdamaxState(_OptStateDataclassBase):
    """State for `scale_by_adamax` holding step count, first moment `mu`, and infinity norm `nu`.

    Attributes:
        count: Optimization step counter Tensor.
        mu: First moment accumulator PyTree.
        nu: Exponentially weighted infinity-norm accumulator PyTree.
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
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByAdamaxState:
        """Construct `ScaleByAdamaxState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, mu, nu)`.

        Returns:
            Reconstructed `ScaleByAdamaxState`.
        """
        items = list(iterable)
        return cls(
            count=_as_count_tensor(items[0]),
            mu=items[1],
            nu=items[2],
        )


def scale_by_adamax(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> GradientTransformation[ScaleByAdamaxState]:
    """Rescale updates according to the AdaMax algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the infinity-norm moment estimate.
        eps: Small constant added to `|g|` in the denominator to avoid division by zero.

    Returns:
        A `GradientTransformation` implementing AdaMax scaling.
    """

    def init_fn(params: Params) -> ScaleByAdamaxState:
        """Initialize `ScaleByAdamaxState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByAdamaxState`.
        """
        return ScaleByAdamaxState(
            count=asarray(0),
            mu=tree_zeros_like(params),
            nu=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByAdamaxState]:
        """Update `mu` and `nu` and return `(mu / (1 - b1^t)) / nu`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByAdamaxState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(scaled_updates, new_adamax_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByAdamaxState`.
        """
        del params
        if not isinstance(state, ScaleByAdamaxState):
            raise TypeError(
                f"Expected ScaleByAdamaxState in scale_by_adamax, got {type(state)!r}."
            )
        count_inc = state.count + 1
        bc1 = 1.0 - power(asarray(b1, dtype="float32"), count_inc)
        safe_bc1 = where(abs_tensor(bc1) > 1e-12, bc1, 1.0)

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
            """Update infinity moment `max(b2 * v, |g| + eps)`.

            Args:
                v: Previous infinity moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated infinity moment leaf.
            """
            if v is None or g is None:
                return v
            return maximum(b2 * asarray(v), abs_tensor(asarray(g)) + eps)

        def _scale_step(m: TensorLeaf, v: TensorLeaf) -> TensorLeaf:
            """Compute `(m / bc1) / v`.

            Args:
                m: Updated first moment leaf.
                v: Updated infinity moment leaf.

            Returns:
                Scaled update leaf.
            """
            if m is None or v is None:
                return m
            return (asarray(m) / safe_bc1) / asarray(v)

        new_mu = tree_map(_update_mu, state.mu, updates)
        new_nu = tree_map(_update_nu, state.nu, updates)
        out_updates = tree_map(_scale_step, new_mu, new_nu)
        return (
            out_updates,
            ScaleByAdamaxState(count=count_inc, mu=new_mu, nu=new_nu),
        )

    return GradientTransformation(init_fn, update_fn)


def adamax(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The AdaMax optimizer (a variant of Adam based on the infinity norm).

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the exponentially weighted infinity norm.
        eps: Small constant added to denominator for numerical stability.

    Returns:
        A `GradientTransformation` implementing AdaMax.
    """
    return chain(
        scale_by_adamax(b1=b1, b2=b2, eps=eps),
        scale_by_learning_rate(learning_rate),
    )


__all__ = [
    "ScaleByAdamaxState",
    "adamax",
    "scale_by_adamax",
]
