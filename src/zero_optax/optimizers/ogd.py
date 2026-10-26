"""Optimistic Gradient Descent (OGD) optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zero_optax._tree import (
    asarray,
    evaluate_schedule,
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
class ScaleByOptimisticGradientState(_OptStateDataclassBase):
    """State for `scale_by_optimistic_gradient` holding step count and previous gradient.

    Attributes:
        count: Optimization step counter Tensor.
        trace: Previous gradient PyTree (initialized to initial parameters/zeros).
    """

    count: Tensor
    trace: Updates
    _fields: tuple[str, ...] = ("count", "trace")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'count'` and `'trace'`.
        """
        return {"count": self.count, "trace": self.trace}

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByOptimisticGradientState:
        """Construct `ScaleByOptimisticGradientState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, trace)`.

        Returns:
            Reconstructed `ScaleByOptimisticGradientState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]), trace=items[1])


def scale_by_optimistic_gradient(
    alpha: ScalarOrSchedule = 1.0,
    beta: ScalarOrSchedule = 1.0,
) -> GradientTransformation[ScaleByOptimisticGradientState]:
    """Compute generalized optimistic gradients `(alpha + beta) * g_t - beta * g_{t-1}`.

    Args:
        alpha: Coefficient or `Schedule` for current gradient weight.
        beta: Coefficient or `Schedule` for optimistic negative momentum weight.

    Returns:
        A `GradientTransformation` implementing optimistic gradient scaling.
    """

    def init_fn(params: Params) -> ScaleByOptimisticGradientState:
        """Initialize `ScaleByOptimisticGradientState` with zero trace.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByOptimisticGradientState`.
        """
        return ScaleByOptimisticGradientState(
            count=asarray(0),
            trace=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByOptimisticGradientState]:
        """Compute optimistic gradient step and store current `updates` in `state.trace`.

        Args:
            updates: Current gradient updates PyTree.
            state: Current `ScaleByOptimisticGradientState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(optimistic_updates, new_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByOptimisticGradientState`.
        """
        del params
        if not isinstance(state, ScaleByOptimisticGradientState):
            raise TypeError(
                f"Expected ScaleByOptimisticGradientState, got {type(state)!r}."
            )
        is_step0 = state.count == 0
        a_t = evaluate_schedule(alpha, state.count)
        b_t = evaluate_schedule(beta, state.count)

        def _ogd_leaf(g: TensorLeaf, prev_g: TensorLeaf) -> TensorLeaf:
            """Compute `alpha * g` on step 0 and `(alpha + beta) * g - beta * prev_g` for `t > 0`.

            Args:
                g: Current gradient leaf.
                prev_g: Previous gradient leaf.

            Returns:
                Optimistic gradient Tensor leaf.
            """
            if g is None or prev_g is None:
                return g
            g_arr = asarray(g)
            return where(
                is_step0,
                a_t * g_arr,
                (a_t + b_t) * g_arr - b_t * asarray(prev_g),
            )

        out_updates = tree_map(_ogd_leaf, updates, state.trace)
        return (
            out_updates,
            ScaleByOptimisticGradientState(count=state.count + 1, trace=updates),
        )

    return GradientTransformation(init_fn, update_fn)


def optimistic_gradient_descent(
    learning_rate: ScalarOrSchedule,
    alpha: ScalarOrSchedule = 1.0,
    beta: ScalarOrSchedule = 1.0,
) -> GradientTransformation[tuple[OptState, ...]]:
    """An Optimistic Gradient Descent (OGD) optimizer.

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        alpha: Coefficient or `Schedule` for generalized OGD.
        beta: Coefficient or `Schedule` for generalized OGD negative momentum.

    Returns:
        A `GradientTransformation` implementing Optimistic Gradient Descent.
    """
    return chain(
        scale_by_optimistic_gradient(alpha=alpha, beta=beta),
        scale_by_learning_rate(learning_rate),
    )


__all__ = [
    "ScaleByOptimisticGradientState",
    "optimistic_gradient_descent",
    "scale_by_optimistic_gradient",
]
