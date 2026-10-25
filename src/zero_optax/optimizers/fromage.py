"""Fromage (Frobenius matched gradient descent) optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zero_optax._tree import (
    asarray,
    evaluate_schedule,
    sqrt,
)
from zero_optax._types import (
    Params,
    PyTree,
    ScalarOrSchedule,
    Tensor,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
    _as_count_tensor,
    _OptStateDataclassBase,
)
from zero_optax.transform import (
    AddDecayedWeightsState,
    ScaleByTrustRatioState,
    add_decayed_weights,
    scale,
    scale_by_trust_ratio,
)


@dataclass(frozen=True)
class FromageState(_OptStateDataclassBase):
    """State for the Fromage optimizer.

    Attributes:
        count: Optimization step counter Tensor.
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
    def _make(cls, iterable: Sequence[PyTree]) -> FromageState:
        """Construct `FromageState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count,)`.

        Returns:
            Reconstructed `FromageState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]))


def fromage(
    learning_rate: ScalarOrSchedule,
    min_norm: float = 1e-6,
) -> GradientTransformation[FromageState]:
    """Create the Frobenius matched gradient descent (Fromage) optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        min_norm: Minimum floor clipped onto both parameter and gradient norms.

    Returns:
        A `GradientTransformation` implementing Fromage.
    """
    trust_tx = scale_by_trust_ratio(min_norm=min_norm)

    def init_fn(params: Params) -> FromageState:
        """Initialize `FromageState` with `count = 0`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `FromageState`.
        """
        del params
        return FromageState(count=asarray(0))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, FromageState]:
        """Compute Fromage trust-ratio update and norm-preserving weight decay.

        Args:
            updates: Gradient updates PyTree.
            state: Current `FromageState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(transformed_updates, new_fromage_state)`.

        Raises:
            TypeError: If `state` is not a `FromageState`.
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `fromage` update.")
        if not isinstance(state, FromageState):
            raise TypeError(f"Expected FromageState in fromage, got {type(state)!r}.")
        count = state.count
        lr_t = asarray(evaluate_schedule(learning_rate, count))
        mult = 1.0 / sqrt(1.0 + lr_t * lr_t)
        eff_lr = -lr_t * mult
        decay_factor = mult - 1.0

        trust_updates, _ = trust_tx.update(updates, ScaleByTrustRatioState(), params)
        scaled_updates, _ = scale(eff_lr).update(trust_updates, state, params)
        new_updates, _ = add_decayed_weights(decay_factor).update(
            scaled_updates, AddDecayedWeightsState(count=count), params
        )
        return new_updates, FromageState(count=count + 1)

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "FromageState",
    "fromage",
]
