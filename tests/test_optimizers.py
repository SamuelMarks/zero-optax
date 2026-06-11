"""Tests for optimizers."""

from typing import Any, Dict, Tuple

import ml_switcheroo.jnp as jnp
import ml_switcheroo.ops as jops
import numpy as np
import pytest

from zero_jax.tree_util import tree_map

from zero_optax.optimizers import adagrad, adam, adamw, lamb, lars, lion, sgd
from zero_optax.schedules import constant_schedule

# Mock out tree structure using nested dicts
Params = Dict[str, jnp.ndarray]


def _make_params() -> Params:
    return {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}


def _make_updates() -> Params:
    return {"w": jnp.array([0.1, -0.1]), "b": jnp.array([0.05])}


def test_sgd() -> None:
    """Test SGD optimizer."""
    params = _make_params()
    updates = _make_updates()

    # Vanilla SGD
    opt = sgd(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)

    assert new_updates["w"].shape == (2,)
    np.testing.assert_allclose(new_updates["w"], [-0.01, 0.01])

    # SGD with momentum
    opt_mom = sgd(0.1, momentum=0.9)
    state_mom = opt_mom.init(params)
    new_updates_mom, new_state_mom = opt_mom.update(updates, state_mom, params)

    assert new_updates_mom["w"].shape == (2,)

    # SGD with schedule
    opt_sched = sgd(constant_schedule(0.1))
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)
    np.testing.assert_allclose(new_updates_sched["w"], [-0.01, 0.01])


def test_sgd_nesterov() -> None:
    """Test SGD optimizer with Nesterov momentum."""
    params = _make_params()
    updates = _make_updates()

    opt_nest = sgd(0.1, momentum=0.9, nesterov=True)
    state_nest = opt_nest.init(params)
    new_updates_nest, new_state_nest = opt_nest.update(updates, state_nest, params)

    assert new_updates_nest["w"].shape == (2,)


def test_adagrad() -> None:
    """Test Adagrad optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = adagrad(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)

    assert new_updates["w"].shape == (2,)

    opt_sched = adagrad(constant_schedule(0.1))
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_adam() -> None:
    """Test Adam optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = adam(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)

    assert new_updates["w"].shape == (2,)

    opt_sched = adam(constant_schedule(0.1), nesterov=True)
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_adamw() -> None:
    """Test AdamW optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = adamw(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)
    assert new_updates["w"].shape == (2,)

    # Missing params error
    with pytest.raises(ValueError):
        opt.update(updates, state)

    opt_sched = adamw(
        constant_schedule(0.1), mask=lambda p: tree_map(lambda _: True, p)
    )
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_lamb() -> None:
    """Test LAMB optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = lamb(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)
    assert new_updates["w"].shape == (2,)

    with pytest.raises(ValueError):
        opt.update(updates, state)

    opt_sched = lamb(constant_schedule(0.1), mask={"w": False, "b": True})
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_lars() -> None:
    """Test LARS optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = lars(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)
    assert new_updates["w"].shape == (2,)

    with pytest.raises(ValueError):
        opt.update(updates, state)

    opt_sched = lars(
        constant_schedule(0.1),
        weight_decay_mask=lambda p: tree_map(lambda _: True, p),
        trust_ratio_mask=lambda p: tree_map(lambda _: True, p),
        nesterov=True,
    )
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_lion() -> None:
    """Test Lion optimizer."""
    params = _make_params()
    updates = _make_updates()

    opt = lion(0.1)
    state = opt.init(params)
    new_updates, new_state = opt.update(updates, state, params)
    assert new_updates["w"].shape == (2,)

    with pytest.raises(ValueError):
        opt.update(updates, state)

    opt_sched = lion(constant_schedule(0.1), mask=lambda p: tree_map(lambda _: True, p))
    state_sched = opt_sched.init(params)
    new_updates_sched, _ = opt_sched.update(updates, state_sched, params)


def test_mask_types() -> None:
    """Test different mask types."""
    params = _make_params()
    updates = _make_updates()

    # Callable mask (tree returning)
    def mask_fn(p):
        return tree_map(lambda _: True, p)

    opt_adamw = adamw(0.1, mask=mask_fn)
    opt_adamw.update(updates, opt_adamw.init(params), params)

    opt_lamb = lamb(0.1, mask=mask_fn)
    opt_lamb.update(updates, opt_lamb.init(params), params)

    opt_lars = lars(0.1, weight_decay_mask=mask_fn, trust_ratio_mask=mask_fn)
    opt_lars.update(updates, opt_lars.init(params), params)

    opt_lars2 = lars(
        0.1,
        weight_decay_mask={"w": True, "b": False},
        trust_ratio_mask={"w": False, "b": True},
    )
    opt_lars2.update(updates, opt_lars2.init(params), params)

    opt_lion = lion(0.1, mask=mask_fn)
    opt_lion.update(updates, opt_lion.init(params), params)


def test_dict_masks() -> None:
    """Test dictionary masks to get coverage up."""
    params = _make_params()
    updates = _make_updates()

    opt_adamw = adamw(0.1, mask={"w": True, "b": False})
    opt_adamw.update(updates, opt_adamw.init(params), params)

    opt_lion = lion(0.1, mask={"w": True, "b": False})
    opt_lion.update(updates, opt_lion.init(params), params)

    opt_lion2 = lion(0.1, mask=True)  # boolean directly
    opt_lion2.update(updates, opt_lion2.init(params), params)

    opt_adamw2 = adamw(0.1, mask=True)
    opt_adamw2.update(updates, opt_adamw2.init(params), params)

    opt_lamb2 = lamb(0.1, mask=False)
    opt_lamb2.update(updates, opt_lamb2.init(params), params)
    opt_adamw_nest = adamw(0.1, nesterov=True)
    opt_adamw_nest.update(updates, opt_adamw_nest.init(params), params)


def test_combine_chain() -> None:
    from zero_optax.combine import chain

    params = _make_params()
    updates = _make_updates()

    opt_chain = chain(sgd(0.1), sgd(0.2))
    state = opt_chain.init(params)
    new_updates, new_state = opt_chain.update(updates, state, params)
    assert new_updates["w"].shape == (2,)
