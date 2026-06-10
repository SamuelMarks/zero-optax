import numpy as np
from zero_optax.schedules.base import (
    constant_schedule,
    linear_schedule,
    polynomial_schedule,
    exponential_decay,
    cosine_decay_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    join_schedules,
)
from zero_optax.schedules.warmup import (
    warmup_constant_schedule,
    warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule,
)
from zero_optax.schedules.cycle import (
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
)
from zero_optax.schedules.inject import (
    inject_hyperparams,
    inject_stateful_hyperparams,
)


# old import bypass
def dummy_import(*args):
    pass


dummy_import(
    constant_schedule,
    linear_schedule,
    polynomial_schedule,
    exponential_decay,
    cosine_decay_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    join_schedules,
    warmup_constant_schedule,
    warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule,
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
    inject_hyperparams,
    inject_stateful_hyperparams,
)


def test_base_schedules():
    assert linear_schedule(1.0, 0.0, 10, transition_begin=5)(0) == 1.0
    assert polynomial_schedule(1.0, 0.0, 2, 10, transition_begin=5)(0) == 1.0
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(15) == 0.1

    assert constant_schedule(1.0)(10) == 1.0
    assert linear_schedule(1.0, 0.0, 10)(0) == 1.0
    assert linear_schedule(1.0, 0.0, 10)(5) == 0.5
    assert linear_schedule(1.0, 0.0, 10)(10) == 0.0
    assert linear_schedule(1.0, 0.0, 10)(15) == 0.0

    assert polynomial_schedule(1.0, 0.0, 2, 10)(0) == 1.0
    assert polynomial_schedule(1.0, 0.0, 2, 10)(5) == 0.25
    assert polynomial_schedule(1.0, 0.0, 2, 10)(10) == 0.0
    assert polynomial_schedule(1.0, 0.0, 2, 10)(15) == 0.0

    assert exponential_decay(1.0, 10, 0.5)(0) == 1.0
    assert np.allclose(exponential_decay(1.0, 10, 0.5)(10), 0.5)
    assert np.allclose(exponential_decay(1.0, 10, 0.5)(5), 1.0 * (0.5**0.5))
    assert exponential_decay(1.0, 10, 0.5, staircase=True)(5) == 1.0
    assert exponential_decay(1.0, 10, 0.5, end_value=0.8)(10) == 0.8
    assert exponential_decay(1.0, 10, 2.0, end_value=1.5)(10) == 1.5

    assert cosine_decay_schedule(1.0, 10)(0) == 1.0
    assert cosine_decay_schedule(1.0, 10)(10) == 0.0
    assert cosine_decay_schedule(1.0, 10)(20) == 0.0
    assert np.allclose(cosine_decay_schedule(1.0, 10, alpha=0.1)(10), 0.1)

    assert piecewise_constant_schedule(1.0)(5) == 1.0
    assert piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.1})(0) == 1.0
    assert piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.1})(5) == 0.5
    assert piecewise_constant_schedule(1.0, {5: 0.5, 10: 0.2})(10) == 0.1

    assert piecewise_interpolate_schedule("linear", 1.0)(5) == 1.0
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(0) == 1.0
    assert np.allclose(
        piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(2.5), 0.75
    )
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(10) == 0.1
    assert piecewise_interpolate_schedule("none", 1.0, {5: 0.5, 10: 0.1})(7.5) == 0.5


def test_join_schedules():
    s1 = constant_schedule(1.0)
    s2 = constant_schedule(2.0)
    assert join_schedules([s1, s2], [10])(5) == 1.0
    assert join_schedules([s1, s2], [10])(15) == 2.0


def test_warmup_schedules():
    assert warmup_constant_schedule(0.0, 1.0, 10)(5) == 0.5
    assert warmup_constant_schedule(0.0, 1.0, 10)(15) == 1.0

    assert warmup_cosine_decay_schedule(0.0, 1.0, 10, 20)(5) == 0.5
    assert warmup_cosine_decay_schedule(0.0, 1.0, 10, 20)(10) == 1.0
    assert warmup_cosine_decay_schedule(0.0, 1.0, 10, 20)(30) == 0.0

    assert warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(5) == 0.5
    assert warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(10) == 1.0
    assert np.allclose(
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5)(30), 0.5
    )
    assert np.allclose(
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, staircase=True)(20),
        1.0,
    )
    assert np.allclose(
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, end_value=0.8)(30), 0.8
    )
    assert np.allclose(
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 2.0, end_value=1.5)(30), 1.5
    )


def test_cycle_schedules():
    assert np.allclose(cosine_onecycle_schedule(100, 1.0)(0), 1.0 / 25.0)
    assert cosine_onecycle_schedule(100, 1.0)(30) == 1.0
    assert np.allclose(cosine_onecycle_schedule(100, 1.0)(100), (1.0 / 25.0) / 1e4)

    assert np.allclose(linear_onecycle_schedule(100, 1.0)(0), 1.0 / 25.0)
    assert linear_onecycle_schedule(100, 1.0)(30) == 1.0
    assert np.allclose(
        linear_onecycle_schedule(100, 1.0)(57),
        1.0 + (57 - 30) / 55 * (1.0 / 25.0 - 1.0),
    )
    assert np.allclose(linear_onecycle_schedule(100, 1.0)(100), (1.0 / 25.0) / 1e4)

    sgdr = sgdr_schedule(
        [{"init_value": 1.0, "decay_steps": 10}, {"init_value": 1.0, "decay_steps": 20}]
    )
    assert sgdr(0) == 1.0
    assert sgdr(10) == 1.0
    assert sgdr(20) == 0.5
    assert sgdr(30) == 0.0
    assert sgdr_schedule([])(5) == 0.0


def test_inject_schedules():
    assert inject_hyperparams(None) is None
    assert inject_stateful_hyperparams(None) is None


def test_warmup_exponential_branch():
    assert (
        warmup_exponential_decay_schedule(0.0, 1.0, 10, 20, 0.5, transition_begin=5)(12)
        == 1.0
    )


def test_inject_wrappers():
    # test inner_factory not None branch
    assert inject_hyperparams(lambda x: x)(5) == 5
    assert inject_stateful_hyperparams(lambda x: x)(5) == 5


def test_onecycle_fall():
    # test step > warmup_steps + decay_steps
    assert linear_onecycle_schedule(100, 1.0)(90) < 1.0


def test_exponential_decay_branches():
    # test decay_rate <= 0
    assert exponential_decay(1.0, 10, -0.5)(5) == 1.0
    # test step < transition_begin inside schedule where val gets replaced
    assert exponential_decay(1.0, 10, 0.5, transition_begin=5)(2) == 1.0


def test_piecewise_interpolate_none():
    assert piecewise_interpolate_schedule("none", 1.0, {5: 0.5, 10: 0.1})(2) == 1.0


def test_piecewise_interpolate_unreachable():
    # step is >= last boundary will hit line 171
    # step < boundaries will hit line 169
    # The return at the very end of loop is practically unreachable if step < last boundary but it's a fallback.
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(10) == 0.1


def test_interpolate_out_of_bounds():
    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(15) == 0.1


def test_interpolate_out_of_bounds_high():
    from zero_optax.schedules.schedule import piecewise_interpolate_schedule

    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(15) == 0.1


def test_interpolate_mid_bounds():
    from zero_optax.schedules.schedule import piecewise_interpolate_schedule

    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(10) == 0.1


def test_schedule_178():
    from zero_optax.schedules.schedule import piecewise_interpolate_schedule

    assert piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 10: 0.1})(15) == 0.1
