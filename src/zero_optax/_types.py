"""Strict type aliases and structural protocols for zero-optax."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir import DType as IrDType
from zero_chex.types import PyTree as ChexPyTree
from zero_flax.core import FrozenEntry
from zero_flax.nnx.state import StateEntry
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec
from zero_orbax.core import SerializedArray
from zero_orbax.types import PyTree as OrbaxPyTree


@runtime_checkable
class DTypeLikeProtocol(Protocol):
    """Protocol for objects exposing a canonical data type name."""

    @property
    def name(self) -> str:
        """Return the canonical string name of the data type.

        Returns:
            Data type name string.
        """
        ...


@runtime_checkable
class ShapeDtypeProtocol(Protocol):
    """Protocol for array-like objects exposing `shape` and `dtype` attributes."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the array as a tuple of integers.

        Returns:
            Tuple of dimension sizes.
        """
        ...

    @property
    def dtype(
        self,
    ) -> Union[
        DType,
        IrDType,
        DTypeLikeProtocol,
        str,
        type[Union[int, float, complex, bool]],
        Tensor,
        ShapeDtypeProtocol,
        JaxArrayLikeSpec,
        None,
    ]:
        """Return the element data type of the array.

        Returns:
            Element data type descriptor.
        """
        ...


@runtime_checkable
class ArrayLikeProtocol(ShapeDtypeProtocol, Protocol):
    """Structural protocol for array-like objects exposing `.tolist()`, `.shape`, and `.dtype` (e.g., `zero_jax` arrays)."""

    def tolist(self) -> Union[ArrayLike, JaxArrayLikeSpec, OrbaxPyTree]:
        """Return the array contents as a Python scalar or nested list.

        Returns:
            Nested list or scalar value.
        """
        ...


@runtime_checkable
class ToListArrayProtocol(ShapeDtypeProtocol, Protocol):
    """Structural protocol for array objects exposing `.to_list()`, `.shape`, and `.dtype` (e.g., `zero_chex` `ZeroArray`)."""

    def to_list(self) -> Sequence[Union[int, float, complex, bool]]:
        """Return the flattened scalar elements of the array.

        Returns:
            Sequence of scalar values.
        """
        ...


@runtime_checkable
class VariableLeafProtocol(Protocol):
    """Structural protocol for `zero_flax.nnx.Variable` leaves (`Param`, `BatchStat`, `LoRAParam`)."""

    def __init__(self, value: Union[PyTree, ChexPyTree, None] = None) -> None:
        """Initialize a variable leaf container with `value`.

        Args:
            value: Initial value to store in the variable leaf.
        """
        ...

    @property
    def value(self) -> ArrayLike:
        """Return the underlying array or scalar value stored in the variable.

        Returns:
            Underlying parameter value.
        """
        ...

    def copy(self) -> VariableLeafProtocol:
        """Return an independent copy of the variable container.

        Returns:
            Copied variable container.
        """
        ...

    def replace(self) -> VariableLeafProtocol:
        """Return a new variable container with updated value or metadata.

        Returns:
            Updated variable container.
        """
        ...


CustomMappingChild = Union["PyTree", StateEntry, FrozenEntry, ChexPyTree]


@runtime_checkable
class CustomMappingPyTreeProtocol(Protocol):
    """Structural protocol for custom mapping PyTree containers (`zero_flax.nnx.State`, `zero_flax.core.FrozenDict`)."""

    def __init__(self, mapping: Mapping[str, PyTree] | None = None) -> None:
        """Initialize the mapping container from a dictionary of child entries.

        Args:
            mapping: Optional mapping of string keys to child PyTrees.
        """
        ...

    def keys(self) -> Iterable[str]:
        """Return an iterable over the mapping keys.

        Returns:
            Iterable of string keys.
        """
        ...

    def __getitem__(self, key: str) -> CustomMappingChild:
        """Retrieve a child PyTree value by key.

        Args:
            key: String key to look up.

        Returns:
            Child PyTree value.
        """
        ...

    def __len__(self) -> int:
        """Return the number of entries in the mapping.

        Returns:
            Number of entries.
        """
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys in the mapping.

        Returns:
            Iterator over string keys.
        """
        ...


@runtime_checkable
class MappableDataclassProtocol(Protocol):
    """Structural protocol for `zero_chex` mappable dataclass PyTree nodes."""

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of field names in the dataclass.

        Returns:
            Tuple of field names.
        """
        ...

    def __chex_field_values__(self) -> tuple[CustomMappingChild, ...]:
        """Return the ordered tuple of field values in the dataclass.

        Returns:
            Tuple of field values.
        """
        ...


@runtime_checkable
class OptStateProtocol(Protocol):
    """Structural protocol for all `zero-optax` state containers (`NamedTuple` or `@dataclass`)."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """Return the tuple of field names defined on the state class.

        Returns:
            Tuple of field names.
        """
        ...

    def _asdict(self) -> dict[str, PyTree]:
        """Return an ordered mapping from field names to their current state values.

        Returns:
            Dictionary mapping field names to state values.
        """
        ...

    @classmethod
    def _make(
        cls, iterable: Sequence[PyTree]
    ) -> Union[tuple[PyTree, ...], Mapping[str, PyTree]]:
        """Construct a new state instance from a sequence of field values matching `_fields`.

        Args:
            iterable: Sequence of field values matching `_fields`.

        Returns:
            Reconstructed state container instance.
        """
        ...


@runtime_checkable
class NamedTupleProtocol(OptStateProtocol, Protocol):
    """Structural protocol for identifying NamedTuple instances without reflection."""

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> tuple[PyTree, ...]:
        """Construct a new NamedTuple instance from a sequence of field values.

        Args:
            iterable: Sequence of field values matching `_fields`.

        Returns:
            Reconstructed NamedTuple instance (which is a tuple subclass).
        """
        ...

    def __len__(self) -> int:
        """Return the number of fields in the NamedTuple.

        Returns:
            Number of fields.
        """
        ...

    def __iter__(self) -> Iterator[PyTree]:
        """Iterate over the field values in the NamedTuple.

        Returns:
            Iterator over field values.
        """
        ...


Array = Tensor
Scalar = Union[int, float, bool]
ArrayLike = Union[
    Tensor,
    Scalar,
    ArrayLikeProtocol,
    ToListArrayProtocol,
    ShapeDtypeProtocol,
    VariableLeafProtocol,
    Sequence["ArrayLike"],
]
DTypeLike = Union[
    DType,
    IrDType,
    DTypeLikeProtocol,
    str,
    type[int],
    type[float],
    type[bool],
    None,
]

TensorLeaf = Union[
    Tensor,
    float,
    int,
    bool,
    None,
    ArrayLikeProtocol,
    ToListArrayProtocol,
    ShapeDtypeProtocol,
    VariableLeafProtocol,
]
PyTree = Union[
    TensorLeaf,
    tuple["PyTree", ...],
    list["PyTree"],
    Mapping[str, "PyTree"],
    CustomMappingPyTreeProtocol,
    MappableDataclassProtocol,
]
Params = PyTree
Updates = PyTree

StateDictTree = Union[
    int,
    float,
    bool,
    None,
    SerializedArray,
    tuple["StateDictTree", ...],
    list["StateDictTree"],
    dict[str, "StateDictTree"],
]

MaskTree = Union[
    bool,
    VariableLeafProtocol,
    tuple["MaskTree", ...],
    list["MaskTree"],
    Mapping[str, "MaskTree"],
    CustomMappingPyTreeProtocol,
]
MaskOrFn = Union[MaskTree, Callable[[Params], MaskTree], None]

Schedule = Callable[[Union[int, Tensor]], Union[float, Tensor]]
ScalarOrSchedule = Union[float, int, Tensor, Schedule]
ReduceFn = Callable[[Tensor], Tensor]


__all__ = [
    "Array",
    "ArrayLike",
    "ArrayLikeProtocol",
    "CustomMappingChild",
    "CustomMappingPyTreeProtocol",
    "DType",
    "DTypeLike",
    "DTypeLikeProtocol",
    "MappableDataclassProtocol",
    "MaskOrFn",
    "MaskTree",
    "NamedTupleProtocol",
    "OptStateProtocol",
    "Params",
    "PyTree",
    "ReduceFn",
    "Scalar",
    "ScalarOrSchedule",
    "Schedule",
    "ShapeDtypeProtocol",
    "StateDictTree",
    "Tensor",
    "TensorLeaf",
    "ToListArrayProtocol",
    "Updates",
    "VariableLeafProtocol",
]
