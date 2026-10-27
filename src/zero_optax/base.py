"""Base classes, states, and protocols for zero-optax gradient transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, TypeVar, Union

from typing_extensions import TypeGuard
from zero_jax import tree_util as jax_tree_util
from zero_orbax.checkpoint import Future as OrbaxFuture
from zero_orbax.core import SerializedArray
from zero_orbax.types import PyTree as OrbaxPyTree

from zero_optax._tree import (
    _coerce_pytree_child,
    _rebuild_mapping,
    _replace_variable_leaf,
    asarray,
    tree_zeros_like,
)
from zero_optax._types import (
    Array,
    ArrayLikeProtocol,
    CustomMappingPyTreeProtocol,
    MappableDataclassProtocol,
    MaskedNode,
    OptStateProtocol,
    Params,
    PyTree,
    Scalar,
    ScalarOrSchedule,
    Schedule,
    StateDictTree,
    Tensor,
    ToListArrayProtocol,
    Updates,
    VariableLeafProtocol,
)

_REGISTERED_OPT_STATE_CLASSES: list[type[_OptStateDataclassBase]] = []


def register_opt_state_pytree(cls: type[_OptStateDataclassBase]) -> None:
    """Register an `_OptStateDataclassBase` subclass with `zero_jax.tree_util.register_pytree_with_keys_class`.

    Args:
        cls: Concrete `_OptStateDataclassBase` class to register.
    """
    if cls not in _REGISTERED_OPT_STATE_CLASSES:
        _REGISTERED_OPT_STATE_CLASSES.append(cls)
    jax_tree_util.register_pytree_with_keys_class(cls)


class _OptStateDataclassBase(Mapping[str, PyTree]):
    """Base class for dataclass-based `OptState` containers providing mapping and PyTree protocols."""

    _fields: tuple[str, ...] = ()

    def __init_subclass__(cls) -> None:
        """Automatically register every `_OptStateDataclassBase` subclass as a custom PyTree node."""
        super().__init_subclass__()
        register_opt_state_pytree(cls)

    @property
    def __orbax_state_type__(self) -> str:
        """Return the canonical Orbax state type identifier for this optimizer state.

        Returns:
            Class name string of the concrete `OptState` type.
        """
        return type(self).__name__

    def _asdict(self) -> dict[str, PyTree]:
        """Return an ordered mapping from field names to their current state values.

        Returns:
            Dictionary mapping field names to state values.
        """
        return {}

    def tree_flatten_with_keys(
        self,
    ) -> tuple[list[tuple[jax_tree_util.GetAttrKey, PyTree]], tuple[str, ...]]:
        """Flatten this optimizer state into `(GetAttrKey(field), value)` pairs and field names.

        Returns:
            Tuple of `(key_value_pairs, field_names_tuple)`.
        """
        return (
            [
                (jax_tree_util.GetAttrKey(k), v)
                for k, v in zip(
                    self.__chex_mappable_fields__(),
                    self.__chex_field_values__(),
                )
            ],
            self.__chex_mappable_fields__(),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[str, ...] | None,
        children: Sequence[PyTree],
    ) -> _OptStateDataclassBase:
        """Reconstruct this optimizer state from `aux_data` and `children`.

        Args:
            aux_data: Field names tuple (unused by `_make`).
            children: Reconstructed child values in field order.

        Returns:
            Reconstructed `_OptStateDataclassBase` instance.
        """
        del aux_data
        return cls._make(list(children))

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of field names for `zero_chex` PyTree traversal.

        Returns:
            Tuple of field names.
        """
        return self._fields

    def __chex_field_values__(self) -> tuple[PyTree, ...]:
        """Return the ordered tuple of field values for `zero_chex` PyTree traversal.

        Returns:
            Tuple of field values matching `_fields`.
        """
        as_dict = self._asdict()
        return tuple(as_dict[k] for k in self._fields)

    def __getitem__(self, key: str) -> PyTree:
        """Retrieve a state field value by field name.

        Args:
            key: Field name in `_fields`.

        Returns:
            Value of the requested state field.
        """
        return self._asdict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the field names in `_fields`.

        Returns:
            Iterator over field name strings.
        """
        return iter(self._fields)

    def __len__(self) -> int:
        """Return the number of state fields.

        Returns:
            Number of fields in `_fields`.
        """
        return len(self._fields)

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> _OptStateDataclassBase:
        """Construct a new state dataclass instance from a sequence of field values.

        Args:
            iterable: Sequence of field values matching `_fields`.

        Returns:
            Reconstructed `_OptStateDataclassBase` instance.
        """
        del iterable
        return cls()


register_opt_state_pytree(_OptStateDataclassBase)


def _as_count_tensor(val: PyTree) -> Tensor:
    """Coerce a deserialized count field into a `Tensor`.

    Args:
        val: Raw count field value.

    Returns:
        Count `Tensor`.
    """
    if isinstance(val, Tensor):
        return val
    return asarray(
        val
        if isinstance(val, (int, float, bool, ArrayLikeProtocol, ToListArrayProtocol))
        else 0
    )


class EmptyState(NamedTuple):
    """State for stateless gradient transformations."""


class TraceState(NamedTuple):
    """State for momentum/trace accumulation transformations.

    Attributes:
        trace: Rolling momentum trace PyTree matching parameters structure.
    """

    trace: Updates


@dataclass(frozen=True)
class ScaleByScheduleState(_OptStateDataclassBase):
    """State holding the integer step count for schedule-based scaling.

    Attributes:
        count: Current optimization step count Tensor.
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
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByScheduleState:
        """Construct `ScaleByScheduleState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count,)`.

        Returns:
            Reconstructed `ScaleByScheduleState`.
        """
        items = list(iterable)
        return cls(count=_as_count_tensor(items[0]))


class ScaleState(NamedTuple):
    """Empty state for constant step-size scaling transformations."""


OptState = PyTree

StateT_co = TypeVar("StateT_co", bound=OptState, covariant=True)


class TransformInitFn(Protocol[StateT_co]):
    """Protocol for a callable that initializes optimizer state from model parameters."""

    def __call__(self, params: Params) -> StateT_co:
        """Initialize and return the optimizer state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized optimizer state.
        """
        ...


class TransformUpdateFn(Protocol[StateT_co]):
    """Protocol for a callable that transforms gradients and updates optimizer state."""

    def __call__(
        self,
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, StateT_co]:
        """Transform gradient `updates` and advance `state`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional current model parameters PyTree.

        Returns:
            A tuple `(transformed_updates, new_state)`.
        """
        ...


ExtraArgValue = Union[
    PyTree,
    Callable[[Params], Union[Scalar, Tensor]],
    Callable[[Params, float], Union[Scalar, Tensor]],
    Callable[[Params, PyTree], Union[Scalar, Tensor]],
    Callable[[Params, PyTree, PyTree], Union[Scalar, Tensor]],
]


class TransformUpdateExtraArgsFn(Protocol[StateT_co]):
    """Protocol for an update function accepting extra keyword arguments."""

    def __call__(
        self,
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, StateT_co]:
        """Transform gradient `updates` with optional extra keyword arguments.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional current model parameters PyTree.
            **extra_args: Additional keyword arguments passed to the transformation.

        Returns:
            A tuple `(transformed_updates, new_state)`.
        """
        ...


class GradientTransformation(Generic[StateT_co]):
    """A pair of pure functions `(init, update)` implementing a gradient transformation.

    Attributes:
        init: Pure function initializing the optimizer state from parameters.
        update: Pure function transforming updates and advancing the optimizer state.
    """

    init: TransformInitFn[StateT_co]
    update: TransformUpdateFn[StateT_co]
    _fields: tuple[str, ...] = ("init", "update")

    def __init__(
        self,
        init: TransformInitFn[StateT_co],
        update: TransformUpdateFn[StateT_co],
        *args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Initialize a `GradientTransformation` from `init` and `update` callables.

        Args:
            init: Pure function initializing the optimizer state from parameters.
            update: Pure function transforming updates and advancing the optimizer state.
            *args: Optional extra positional arguments (unused, for NamedTuple signature parity).
            **kwargs: Optional extra keyword arguments (unused, for NamedTuple signature parity).
        """
        del args, kwargs
        self.init = init
        self.update = update

    def __iter__(
        self,
    ) -> Iterator[TransformInitFn[StateT_co] | TransformUpdateFn[StateT_co]]:
        """Iterate over `(init, update)` for tuple unpacking compatibility.

        Returns:
            Iterator yielding `init` and `update`.
        """
        yield self.init
        yield self.update


def _is_init_fn(
    fn: TransformUpdateExtraArgsFn[StateT_co] | TransformInitFn[StateT_co] | PyTree,
    has_second_callable: bool,
) -> TypeGuard[TransformInitFn[StateT_co]]:
    """Narrow `fn` to `TransformInitFn[StateT_co]` when a second update callable is supplied.

    Args:
        fn: Candidate init callable.
        has_second_callable: Whether a second update callable was provided.

    Returns:
        True if `has_second_callable` and `fn` is callable.
    """
    return has_second_callable and callable(fn)


def _is_update_extra_fn(
    fn: TransformUpdateExtraArgsFn[StateT_co] | TransformInitFn[StateT_co] | PyTree,
) -> TypeGuard[TransformUpdateExtraArgsFn[StateT_co]]:
    """Narrow `fn` to `TransformUpdateExtraArgsFn[StateT_co]` when `fn` is callable.

    Args:
        fn: Candidate update callable.

    Returns:
        True if `fn` is callable.
    """
    return callable(fn)


class GradientTransformationExtraArgs(GradientTransformation[StateT_co]):
    """Gradient transformation whose `update` accepts extra keyword arguments.

    Attributes:
        init: Pure function initializing the optimizer state from parameters.
        update: Pure function accepting extra keyword arguments during update.
    """

    init: TransformInitFn[StateT_co]
    update: TransformUpdateExtraArgsFn[StateT_co]
    _fields: tuple[str, ...] = ("init", "update")

    def __init__(
        self,
        update: TransformUpdateExtraArgsFn[StateT_co]
        | TransformInitFn[StateT_co]
        | None = None,
        *args: TransformUpdateExtraArgsFn[StateT_co] | PyTree,
        **kwargs: TransformUpdateExtraArgsFn[StateT_co]
        | TransformInitFn[StateT_co]
        | PyTree,
    ) -> None:
        """Initialize a `GradientTransformationExtraArgs` from `(init, update)`.

        Args:
            update: Either `init` (when `update` is passed as the second positional argument in `*args`)
                or `update` when `init` is passed via `**kwargs`.
            *args: Second positional argument `update` callable.
            **kwargs: Optional keyword argument `init` callable.

        Raises:
            TypeError: If both `init` and `update` callables are not provided.
        """
        first_fn: (
            TransformUpdateExtraArgsFn[StateT_co] | TransformInitFn[StateT_co] | PyTree
        ) = kwargs.get("init", update)
        second_fn: (
            TransformUpdateExtraArgsFn[StateT_co] | TransformInitFn[StateT_co] | PyTree
        ) = (
            args[0]
            if len(args) >= 1
            else (update if "init" in kwargs else kwargs.get("update"))
        )
        if _is_init_fn(first_fn, callable(second_fn)) and _is_update_extra_fn(
            second_fn
        ):
            self.init = first_fn
            self.update = second_fn
        else:
            raise TypeError(
                "GradientTransformationExtraArgs requires (init, update) callables."
            )


def identity() -> GradientTransformation[EmptyState]:
    """Stateless identity transformation that leaves gradient updates unchanged.

    Returns:
        A `GradientTransformation` that returns updates unmodified.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for identity transformation.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            An `EmptyState` instance.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Return `updates` unchanged with an empty state.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(updates, EmptyState())`.
        """
        del state, params
        return updates, EmptyState()

    return GradientTransformation(init_fn, update_fn)


def set_to_zero() -> GradientTransformation[EmptyState]:
    """Stateless transformation that sets all gradient updates to zero.

    Returns:
        A `GradientTransformation` that zeroes out all updates.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for set_to_zero transformation.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            An `EmptyState` instance.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Replace all leaves in `updates` with zeros of matching shape and dtype.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(zero_updates, EmptyState())`.
        """
        del state, params
        return tree_zeros_like(updates), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def stateless(
    f: Callable[[Updates, Params | None], Updates],
) -> GradientTransformation[EmptyState]:
    """Create a stateless `GradientTransformation` from an update-like function `f(updates, params)`.

    Args:
        f: Pure callable taking `(updates, params)` and returning transformed `updates`.

    Returns:
        A stateless `GradientTransformation` wrapping `f`.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `stateless`.

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
        """Apply `f(updates, params)` and return `EmptyState()`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(f(updates, params), EmptyState())`.
        """
        del state
        return f(updates, params), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def _tensor_to_nested_list(t: Tensor) -> OrbaxPyTree:
    """Convert a `Tensor` into a nested Python list or scalar matching `t.shape`.

    Args:
        t: Input `Tensor`.

    Returns:
        Scalar (if 0-D) or nested list of scalars matching `t.shape`.
    """
    if t.ndim == 0:
        return t.item()
    return [_tensor_to_nested_list(sub) for sub in t]


def _tensor_to_serialized_array(t: Tensor) -> SerializedArray:
    """Convert a `Tensor` into a `zero_orbax.core.SerializedArray`.

    Args:
        t: Input `Tensor`.

    Returns:
        `SerializedArray` with nested data, shape, and dtype name.
    """
    dt_str = str(t.dtype)
    clean_dt = dt_str[6:] if dt_str.startswith("DType.") else dt_str
    return SerializedArray(
        data=_tensor_to_nested_list(t), shape=t.shape, dtype=clean_dt
    )


def opt_state_to_state_dict(state: OptState) -> StateDictTree:
    """Convert an `OptState` container hierarchy into a serializable dictionary/tuple PyTree.

    Args:
        state: Optimizer state (`OptStateProtocol`, tuple, list, mapping, or leaf).

    Returns:
        Pure dictionary/tuple/array PyTree suitable for `zero-orbax` or `zero-flax` serialization.
    """
    if isinstance(state, MaskedNode):
        return {"__orbax_state_type__": state.__orbax_state_type__}
    if isinstance(state, MappableDataclassProtocol) and not isinstance(
        state, _OptStateDataclassBase
    ):
        return {
            k: opt_state_to_state_dict(_coerce_pytree_child(v))
            for k, v in zip(
                state.__chex_mappable_fields__(), state.__chex_field_values__()
            )
        }
    if isinstance(state, OptStateProtocol):
        raw_map = state._asdict()
        return {k: opt_state_to_state_dict(raw_map[k]) for k in state._fields}
    if isinstance(state, (Mapping, CustomMappingPyTreeProtocol)):
        return {
            str(k): opt_state_to_state_dict(_coerce_pytree_child(state[k]))
            for k in state
        }
    if isinstance(state, tuple):
        return tuple(opt_state_to_state_dict(item) for item in state)
    if isinstance(state, list):
        return [opt_state_to_state_dict(item) for item in state]
    if isinstance(state, VariableLeafProtocol):
        return opt_state_to_state_dict(asarray(state.value))
    if isinstance(state, Tensor):
        return _tensor_to_serialized_array(state)
    if isinstance(state, (int, float, bool, type(None))):
        return state
    return _tensor_to_serialized_array(asarray(state))


def _unwrap_orbax_state_dict(
    state_dict: PyTree | OrbaxPyTree | OrbaxFuture,
) -> PyTree:
    """Normalize a restored `zero-orbax` payload or `Future` into a `zero-optax` `PyTree`.

    Args:
        state_dict: Restored checkpoint payload or `Future`.

    Returns:
        Normalized `zero-optax` `PyTree`.
    """
    raw = state_dict.result() if isinstance(state_dict, OrbaxFuture) else state_dict
    if isinstance(raw, MaskedNode):
        return raw
    if isinstance(raw, OptStateProtocol):
        return raw._asdict()
    if isinstance(
        raw,
        (
            Tensor,
            float,
            int,
            bool,
            type(None),
            ArrayLikeProtocol,
            ToListArrayProtocol,
            VariableLeafProtocol,
            CustomMappingPyTreeProtocol,
        ),
    ) and not isinstance(raw, dict):
        return raw
    if isinstance(raw, Mapping):
        if raw.get("__orbax_state_type__") == "MaskedNode":
            return MaskedNode()
        return {str(k): _unwrap_orbax_state_dict(v) for k, v in raw.items()}
    if isinstance(raw, tuple):
        return tuple(_unwrap_orbax_state_dict(item) for item in raw)
    if isinstance(raw, list):
        return [_unwrap_orbax_state_dict(item) for item in raw]
    return None


def opt_state_from_state_dict(
    template: OptState,
    state_dict: PyTree | OrbaxPyTree | OrbaxFuture,
) -> OptState:
    """Reconstruct a strongly-typed `OptState` from a serialized `state_dict` using `template`.

    Args:
        template: Reference `OptState` structure (e.g., from `tx.init(params)`).
        state_dict: Deserialized PyTree dictionary or tuple from `opt_state_to_state_dict` or `zero-orbax`.

    Returns:
        Reconstructed `OptState` matching the exact classes and leaf types of `template`.

    Raises:
        TypeError: If `state_dict` container structure is incompatible with `template`.
    """
    if isinstance(template, MaskedNode):
        return MaskedNode()
    norm_dict = _unwrap_orbax_state_dict(state_dict)
    if isinstance(template, OptStateProtocol):
        template_map = template._asdict()
        if isinstance(norm_dict, (Mapping, CustomMappingPyTreeProtocol)):
            reconstructed = [
                opt_state_from_state_dict(
                    template_map[k], _coerce_pytree_child(norm_dict[k])
                )
                for k in template._fields
            ]
            return template._make(reconstructed)
        if isinstance(norm_dict, (tuple, list)):
            reconstructed = [
                opt_state_from_state_dict(template_map[k], val)
                for k, val in zip(template._fields, norm_dict)
            ]
            return template._make(reconstructed)
        raise TypeError(
            f"Cannot restore {type(template).__name__} from {type(norm_dict).__name__}."
        )
    if isinstance(template, (Mapping, CustomMappingPyTreeProtocol)) and isinstance(
        norm_dict, (Mapping, CustomMappingPyTreeProtocol)
    ):
        mapped: dict[str, PyTree] = {
            str(k): opt_state_from_state_dict(
                _coerce_pytree_child(template[k]),
                _coerce_pytree_child(norm_dict[k]),
            )
            for k in template
        }
        return _rebuild_mapping(template, mapped)
    if isinstance(template, tuple) and isinstance(norm_dict, (tuple, list)):
        return tuple(
            opt_state_from_state_dict(t_item, s_item)
            for t_item, s_item in zip(template, norm_dict)
        )
    if isinstance(template, list) and isinstance(norm_dict, (tuple, list)):
        return [
            opt_state_from_state_dict(t_item, s_item)
            for t_item, s_item in zip(template, norm_dict)
        ]
    if isinstance(template, VariableLeafProtocol):
        inner_state: PyTree = (
            asarray(norm_dict.value)
            if isinstance(norm_dict, VariableLeafProtocol)
            else norm_dict
        )
        restored_val = opt_state_from_state_dict(asarray(template.value), inner_state)
        return _replace_variable_leaf(template, restored_val)
    if isinstance(template, Tensor) and isinstance(
        norm_dict,
        (Tensor, int, float, bool, ArrayLikeProtocol, ToListArrayProtocol),
    ):
        return asarray(norm_dict, dtype=template.dtype)
    return norm_dict


def register_all_opt_states() -> None:
    """Register `_OptStateDataclassBase` and all concrete `OptState` subclasses in `zero_jax` and `zero_chex`."""
    for cls in list(_REGISTERED_OPT_STATE_CLASSES):
        register_opt_state_pytree(cls)


__all__ = [
    "Array",
    "EmptyState",
    "GradientTransformation",
    "GradientTransformationExtraArgs",
    "OptState",
    "Params",
    "PyTree",
    "ScalarOrSchedule",
    "ScaleByScheduleState",
    "ScaleState",
    "Schedule",
    "TraceState",
    "TransformInitFn",
    "TransformUpdateExtraArgsFn",
    "TransformUpdateFn",
    "Updates",
    "_OptStateDataclassBase",
    "_as_count_tensor",
    "identity",
    "opt_state_from_state_dict",
    "opt_state_to_state_dict",
    "register_all_opt_states",
    "register_opt_state_pytree",
    "set_to_zero",
    "stateless",
]
