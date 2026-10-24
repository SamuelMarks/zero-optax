"""Comprehensive unit tests for `zero_optax.schedules`."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tests.utils import assert_allclose_optax, get_tree_item, to_numpy
from zero_optax._tree import asarray
from zero_optax._types import Tensor
from zero_optax.base import EmptyState, OptState
from zero_optax.optimizers import adamw, sgd
from zero_optax.schedules import (
    InjectHyperparamsState,
    InjectStatefulHyperparamsState,
    constant_schedule,
    cosine_decay_schedule,
    cosine_onecycle_schedule,
    exponential_decay,
    inject_hyperparams,
    inject_stateful_hyperparams,
    join_schedules,
    linear_onecycle_schedule,
    linear_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    polynomial_schedule,
    sgdr_schedule,
    warmup_constant_schedule,
    warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule,
)
from zero_optax.schedules.schedule import _step_to_float


class _MockStatefulSchedule:
    """Mock stateful schedule satisfying `StatefulScheduleProtocol`."""

    def init(self) -> OptState:
        """Initialize the schedule state to scalar Tensor `1.0`.

        Returns:
            Initial state Tensor.
        """
        return asarray(1.0)

    def update(
        self,
        state: OptState,
        **extra_args: float | int | str | bool | Tensor | None,
    ) -> OptState:
        """Double the schedule state Tensor on each step.

        Args:
            state: Current state Tensor.
            **extra_args: Unused extra keyword arguments.

        Returns:
            Updated state Tensor.
        """
        del extra_args
        cur = state if isinstance(state, (int, float, Tensor)) else 1.0
        return asarray(cur) * 2.0

    def __call__(
        self,
        state: OptState,
        **extra_args: float | int | str | bool | Tensor | None,
    ) -> float | int | Tensor:
        """Evaluate the current learning rate from `state`.

        Args:
            state: Current state Tensor.
            **extra_args: Unused extra keyword arguments.

        Returns:
            Learning rate `0.1 * state`.
        """
        del extra_args
        cur = state if isinstance(state, (int, float, Tensor)) else 1.0
        return asarray(cur) * 0.1


def test_base_schedules() -> None:
    """Test `constant_schedule`, `linear_schedule`, `polynomial_schedule`, `exponential_decay`, and `cosine_decay_schedule`."""
    assert constant_schedule(1.0)(10) == 1.0
    assert constant_schedule(1.0)(asarray(10)) == 1.0

    assert linear_schedule(1.0, 0.0, 10, transition_begin=5)(0) == 1.0
    assert linear_schedule(1.0, 0.0, 0)(5) == 1.0
    assert linear_schedule(1.0, 0.0, 10)(0) == 1.0
    assert linear_schedule(1.0, 0.0, 10)(5) == 0.5
    assert linear_schedule(1.0, 0.0, 10)(10) == 0.0
    assert linear_schedule(1.0, 0.0, 10)(15) == 0.0

    assert polynomial_schedule(1.0, 0.0, 2.0, 10, transition_begin=5)(0) == 1.0
    assert polynomial_schedule(1.0, 0.0, 2.0, 0)(5) == 1.0
    assert polynomial_schedule(1.0, 0.0, 2.0, 10)(0) == 1.0
    assert polynomial_schedule(1.0, 0.0, 2.0, 10)(5) == 0.25
    assert polynomial_schedule(1.0, 0.0, 2.0, 10)(10) == 0.0
    assert polynomial_schedule(1.0, 0.0, 2.0, 10)(15) == 0.0

    assert exponential_decay(1.0, 10, 0.5)(0) == 1.0
    assert exponential_decay(1.0, 0, 0.5)(5) == 1.0
    assert exponential_decay(1.0, 10, -0.5)(5) == 1.0
    assert exponential_decay(1.0, 10, 0.5, transition_begin=5)(2) == 1.0
    assert math.isclose(_step_to_float(exponential_decay(1.0, 10, 0.5)(10)), 0.5)
    assert math.isclose(
        _step_to_float(exponential_decay(1.0, 10, 0.5)(5)), 1.0 * (0.5**0.5)
    )
    assert exponential_decay(1.0, 10, 0.5, staircase=True)(5) == 1.0
    assert exponential_decay(1.0, 10, 0.5, end_value=0.8)(10) == 0.8
    assert exponential_decay(1.0, 10, 2.0, end_value=1.5)(10) == 1.5

    assert cosine_decay_schedule(1.0, 10)(0) == 1.0
    assert cosine_decay_schedule(1.0, 10)(10) == 0.0
    assert cosine_decay_schedule(1.0, 10)(20) == 0.0
    assert math.isclose(
        _step_to_float(cosine_decay_schedule(1.0, 10, alpha=0.1, exponent=2.0)(10)),
        0.1,
    )


def test_piecewise_and_join_schedules() -> None:
    """Test `piecewise_constant_schedule`, `piecewise_interpolate_schedule`, and `join_schedules`."""
    assert piecewise_constant_schedule(1.0)(5) == 1.0
    assert piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.1})(0) == 1.0
    assert piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.1})(5) == 0.5
    assert math.isclose(
        _step_to_float(piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.2})(10)),
        0.1,
    )

    assert piecewise_interpolate_schedule("linear", 1.0)(5) == 1.0
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(-1) == 1.0
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(0) == 1.0
    assert math.isclose(
        _step_to_float(
            piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(2)
        ),
        float(
            optax.piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(2)
        ),
        rel_tol=1e-5,
    )
    assert math.isclose(
        _step_to_float(
            piecewise_interpolate_schedule("cosine", 1.0, {5: 0.5, 10: 0.2})(2)
        ),
        float(
            optax.piecewise_interpolate_schedule("cosine", 1.0, {5: 0.5, 10: 0.2})(2)
        ),
        rel_tol=1e-5,
    )
    assert math.isclose(
        _step_to_float(
            piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(10)
        ),
        0.1,
    )
    assert math.isclose(
        _step_to_float(
            piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.2})(15)
        ),
        0.1,
    )
    assert piecewise_interpolate_schedule("none", 1.0, {5: 0.5, 10: 0.2})(7) == 0.5

    with pytest.raises(ValueError, match="interpolate_type"):
        piecewise_interpolate_schedule("cubic", 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        piecewise_interpolate_schedule("linear", 1.0, {5: -0.5})

    s1 = linear_schedule(0.0, 1.0, 10)
    s2 = linear_schedule(1.0, 0.0, 10)
    joined = join_schedules([s1, s2], [10])
    assert joined(5) == 0.5
    assert joined(15) == 0.5
    assert joined(asarray(15)) == 0.5


def test_warmup_and_cycle_schedules() -> None:
    """Test warmup schedules, 1-cycle schedules, and `sgdr_schedule` against `optax`."""
    assert warmup_constant_schedule(0.0, 1.0, 10)(5) == 0.5
    assert warmup_constant_schedule(0.0, 1.0, 10)(15) == 1.0

    z_wcos = warmup_cosine_decay_schedule(0.0, 1.0, 10, 30, end_value=0.1)
    o_wcos = optax.warmup_cosine_decay_schedule(0.0, 1.0, 10, 30, end_value=0.1)
    for step in (0, 5, 10, 20, 30, 40):
        assert math.isclose(
            _step_to_float(z_wcos(step)),
            float(o_wcos(step)),
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
    assert warmup_cosine_decay_schedule(0.0, 0.0, 0, 10)(5) == 0.0

    assert warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(5) == 0.5
    assert warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(10) == 1.0
    assert (
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, transition_begin=5)(12)
        == 1.0
    )
    assert math.isclose(
        _step_to_float(warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(30)),
        0.5,
    )
    assert math.isclose(
        _step_to_float(
            warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, staircase=True)(20)
        ),
        1.0,
    )
    assert math.isclose(
        _step_to_float(
            warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, end_value=0.8)(30)
        ),
        0.8,
    )
    assert math.isclose(
        _step_to_float(
            warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 2.0, end_value=1.5)(30)
        ),
        1.5,
    )

    z_cos1 = cosine_onecycle_schedule(100, 1.0)
    o_cos1 = optax.cosine_onecycle_schedule(100, 1.0)
    for step in (0, 15, 30, 65, 100, 110):
        assert math.isclose(
            _step_to_float(z_cos1(step)),
            float(o_cos1(step)),
            rel_tol=1e-5,
            abs_tol=1e-7,
        )
    with pytest.raises(ValueError, match="transition_steps"):
        cosine_onecycle_schedule(0, 1.0)

    z_lin1 = linear_onecycle_schedule(100, 1.0)
    o_lin1 = optax.linear_onecycle_schedule(100, 1.0)
    for step in (0, 15, 30, 57, 85, 95, 100, 105):
        assert math.isclose(
            _step_to_float(z_lin1(step)),
            float(o_lin1(step)),
            rel_tol=1e-5,
            abs_tol=1e-7,
        )
    with pytest.raises(ValueError, match="transition_steps"):
        linear_onecycle_schedule(0, 1.0)

    cycles = [
        {
            "init_value": 0.0,
            "peak_value": 1.0,
            "warmup_steps": 2,
            "decay_steps": 10,
            "end_value": 0.0,
        },
        {
            "init_value": 0.0,
            "peak_value": 0.5,
            "warmup_steps": 4,
            "decay_steps": 20,
            "end_value": 0.0,
        },
    ]
    z_sgdr = sgdr_schedule(cycles)
    o_sgdr = optax.sgdr_schedule(cycles)
    for step in (0, 1, 2, 6, 10, 12, 14, 20, 30, 35):
        assert math.isclose(
            _step_to_float(z_sgdr(step)),
            float(o_sgdr(step)),
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
    assert sgdr_schedule([])(5) == 0.0
    assert sgdr_schedule(None)(5) == 0.0


def test_inject_hyperparams_and_stateful() -> None:
    """Test `inject_hyperparams` and `inject_stateful_hyperparams` against `optax`."""
    assert inject_hyperparams(None) is None
    assert inject_stateful_hyperparams(None) is None

    with pytest.raises(ValueError, match="static_args"):
        inject_hyperparams(sgd, static_args="not_a_param")
    with pytest.raises(ValueError, match="static_args"):
        inject_stateful_hyperparams(sgd, static_args="not_a_param")

    params = {"w": asarray([1.0, 2.0])}
    updates = {"w": asarray([0.1, -0.1])}
    j_params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    j_updates = {"w": jnp.array([0.1, -0.1], dtype=jnp.float32)}

    z_lr_sched = linear_schedule(0.1, 0.01, 10)
    o_lr_sched = optax.linear_schedule(0.1, 0.01, 10)

    injected_sgd_factory = inject_hyperparams(
        sgd,
        static_args="nesterov",
        hyperparam_dtype="float32",
    )
    assert injected_sgd_factory is not None
    o_injected_sgd_factory = optax.inject_hyperparams(
        optax.sgd,
        static_args="nesterov",
        hyperparam_dtype=jnp.float32,
    )
    tx = injected_sgd_factory(z_lr_sched, momentum=0.9, nesterov=True)
    otx = o_injected_sgd_factory(o_lr_sched, momentum=0.9, nesterov=True)

    state = tx.init(params)
    ostate = otx.init(j_params)
    assert isinstance(state, InjectHyperparamsState)
    assert_allclose_optax(
        state.hyperparams["learning_rate"],
        np.asarray(ostate.hyperparams["learning_rate"]),
    )

    u1, state1 = tx.update(updates, state, params)
    ou1, ostate1 = otx.update(j_updates, ostate, j_params)
    assert isinstance(state1, InjectHyperparamsState)
    assert_allclose_optax(get_tree_item(u1, "w"), np.asarray(ou1["w"]), rtol=1e-4)
    assert_allclose_optax(
        state1.hyperparams["learning_rate"],
        np.asarray(ostate1.hyperparams["learning_rate"]),
    )

    u2, _ = tx.update(updates, state1, params)
    ou2, _ = otx.update(j_updates, ostate1, j_params)
    assert_allclose_optax(get_tree_item(u2, "w"), np.asarray(ou2["w"]), rtol=1e-4)

    u_fallback, _ = tx.update(updates, EmptyState(), params)
    assert_allclose_optax(
        get_tree_item(u_fallback, "w"), np.asarray(ou1["w"]), rtol=1e-4
    )

    injected_adamw_factory = inject_hyperparams(adamw)
    assert injected_adamw_factory is not None
    injected_adamw = injected_adamw_factory(
        z_lr_sched, mask=lambda p: {"w": True}, nesterov=True
    )
    o_injected_adamw = optax.inject_hyperparams(optax.adamw, static_args="mask")(
        o_lr_sched, mask=lambda p: {"w": True}, nesterov=True
    )
    aw_st = injected_adamw.init(params)
    o_aw_st = o_injected_adamw.init(j_params)
    aw_u1, _ = injected_adamw.update(updates, aw_st, params)
    o_aw_u1, _ = o_injected_adamw.update(j_updates, o_aw_st, j_params)
    assert_allclose_optax(
        get_tree_item(aw_u1, "w"), np.asarray(o_aw_u1["w"]), rtol=1e-4, atol=1e-4
    )

    stateful_sgd_factory = inject_stateful_hyperparams(
        sgd,
        static_args=("nesterov",),
        hyperparam_dtype=None,
    )
    assert stateful_sgd_factory is not None
    mock_sched = _MockStatefulSchedule()
    st_tx = stateful_sgd_factory(mock_sched, momentum=0.0, nesterov=False)
    st_state = st_tx.init(params)
    assert isinstance(st_state, InjectStatefulHyperparamsState)
    assert "learning_rate" in st_state.hyperparams_states
    assert_allclose_optax(st_state.hyperparams["learning_rate"], np.array(0.1))

    su1, st_state1 = st_tx.update(updates, st_state, params)
    assert_allclose_optax(get_tree_item(su1, "w"), np.array([-0.01, 0.01]))
    assert isinstance(st_state1, InjectStatefulHyperparamsState)
    su2, _ = st_tx.update(updates, st_state1, params)
    assert_allclose_optax(get_tree_item(su2, "w"), np.array([-0.02, 0.02]))

    su_fallback, _ = st_tx.update(updates, EmptyState(), params)
    assert_allclose_optax(get_tree_item(su_fallback, "w"), np.array([-0.01, 0.01]))

    callable_st_tx = stateful_sgd_factory(z_lr_sched, momentum=0.0, nesterov=False)
    cst_state = callable_st_tx.init(params)
    csu1, _ = callable_st_tx.update(updates, cst_state, params)
    assert_allclose_optax(get_tree_item(csu1, "w"), np.array([-0.01, 0.01]))


def test_utils_error_branches() -> None:
    """Test `get_tree_item` and `to_numpy` branches in `tests/utils.py`."""
    assert to_numpy([1.0, 2.0]).shape == (2,)
    with pytest.raises(TypeError, match="Expected dict PyTree"):
        get_tree_item(asarray([1.0]), "w")
    with pytest.raises(TypeError, match="Expected Tensor"):
        get_tree_item({"w": None}, "w")
