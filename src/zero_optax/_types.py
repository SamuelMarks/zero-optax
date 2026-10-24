"""Strict type aliases and structural protocols for zero-optax."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor

Array = Tensor
Scalar = Union[int, float, bool]
ArrayLike = Union[Tensor, Scalar, Sequence["ArrayLike"]]
DTypeLike = Union[DType, str, type[int], type[float], type[bool], None]

TensorLeaf = Union[Tensor, float, int, bool, None]
PyTree = Union[
    TensorLeaf,
    tuple["PyTree", ...],
    list["PyTree"],
    Mapping[str, "PyTree"],
]
Params = PyTree
Updates = PyTree

MaskTree = Union[
    bool,
    tuple["MaskTree", ...],
    list["MaskTree"],
    Mapping[str, "MaskTree"],
]
MaskOrFn = Union[MaskTree, Callable[[Params], MaskTree], None]

Schedule = Callable[[Union[int, Tensor]], Union[float, Tensor]]
ScalarOrSchedule = Union[float, int, Tensor, Schedule]
ReduceFn = Callable[[Tensor], Tensor]


@runtime_checkable
class NamedTupleProtocol(Protocol):
    """Structural protocol for identifying NamedTuple instances without reflection.

    Attributes:
        _fields: Tuple of field names defined on the NamedTuple class.
    """

    _fields: tuple[str, ...]


__all__ = [
    "Array",
    "ArrayLike",
    "DType",
    "DTypeLike",
    "MaskOrFn",
    "MaskTree",
    "NamedTupleProtocol",
    "Params",
    "PyTree",
    "ReduceFn",
    "Scalar",
    "ScalarOrSchedule",
    "Schedule",
    "Tensor",
    "TensorLeaf",
    "Updates",
]
