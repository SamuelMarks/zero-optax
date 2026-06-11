"""Base classes and types for zero-optax."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
from typing import Protocol
from zero_jax import Array

PyTree = Any
Params = PyTree
Updates = PyTree
OptState = PyTree


class TransformInitFn(Protocol):
    """A callable type for the `init` step of a `GradientTransformation`."""

    def __call__(self, params: Params) -> OptState:
        """Initializes the optimizer state."""
        ...


class TransformUpdateFn(Protocol):
    """A callable type for the `update` step of a `GradientTransformation`."""

    def __call__(
        self, updates: Updates, state: OptState, params: Optional[Params] = None
    ) -> Tuple[Updates, OptState]:
        """Transforms the updates and updates the optimizer state."""
        ...


class GradientTransformation(NamedTuple):
    """A pair of pure functions implementing a gradient transformation."""

    init: TransformInitFn
    update: TransformUpdateFn
