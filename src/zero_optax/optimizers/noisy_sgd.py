"""Noisy SGD and SignSGD optimizers."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import asarray, sign, tree_map
from zero_optax._types import (
    ArrayLike,
    Params,
    ScalarOrSchedule,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    EmptyState,
    GradientTransformation,
    OptState,
)
from zero_optax.combine import chain
from zero_optax.transform import add_noise, scale_by_learning_rate


class ScaleBySignState(NamedTuple):
    """Empty state for `scale_by_sign`."""


def scale_by_sign() -> GradientTransformation[EmptyState]:
    """Turn updates into their element-wise signs (`sign(g)`).

    Returns:
        A stateless `GradientTransformation` mapping each leaf to `sign(g)`.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `scale_by_sign`.

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
        """Compute `sign(u)` leaf-wise.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(sign_updates, EmptyState())`.
        """
        del state, params

        def _sgn(u: TensorLeaf) -> TensorLeaf:
            """Return `sign(u)` for leaf `u`.

            Args:
                u: Update leaf.

            Returns:
                Sign Tensor leaf.
            """
            return None if u is None else sign(asarray(u))

        return tree_map(_sgn, updates), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def sign_sgd(
    learning_rate: ScalarOrSchedule,
) -> GradientTransformation[tuple[OptState, ...]]:
    """A variant of SGD using only the signs of the gradient components (`-lr * sign(g)`).

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.

    Returns:
        A `GradientTransformation` implementing SignSGD.
    """
    return chain(
        scale_by_sign(),
        scale_by_learning_rate(learning_rate),
    )


def noisy_sgd(
    learning_rate: ScalarOrSchedule,
    eta: float = 0.01,
    gamma: float = 0.55,
    key: ArrayLike | int | None = None,
    *,
    seed: int | None = None,
) -> GradientTransformation[tuple[OptState, ...]]:
    """A variant of SGD with annealed Gaussian gradient noise.

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        eta: Initial variance of the Gaussian noise.
        gamma: Decay exponent controlling noise annealing `(1 + t) ** (-gamma)`.
        key: Optional PRNG key array or integer seed.
        seed: Optional deprecated integer seed keyword argument.

    Returns:
        A `GradientTransformation` implementing Noisy SGD.
    """
    return chain(
        scale_by_learning_rate(learning_rate),
        add_noise(eta=eta, gamma=gamma, key=key, seed=seed),
    )


__all__ = [
    "ScaleBySignState",
    "noisy_sgd",
    "scale_by_sign",
    "sign_sgd",
]
