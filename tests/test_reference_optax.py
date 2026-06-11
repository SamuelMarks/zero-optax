import pytest
import optax
import zero_optax as zoptax
import numpy as np


@pytest.mark.skip(reason="Not yet implemented in zero-optax")
def test_optax_sgd():
    opt_r = optax.sgd(0.1)
    opt_z = zoptax.sgd(0.1)
    # Testing initialization
    params_r = {"w": np.array([1.0, 2.0])}
    params_z = {"w": np.array([1.0, 2.0])}

    state_r = opt_r.init(params_r)
    state_z = opt_z.init(params_z)

    # Testing update
    grads = {"w": np.array([0.1, 0.2])}
    updates_r, new_state_r = opt_r.update(grads, state_r, params_r)
    updates_z, new_state_z = opt_z.update(grads, state_z, params_z)

    np.testing.assert_allclose(updates_z["w"], updates_r["w"])


@pytest.mark.skip(reason="Not yet implemented in zero-optax")
def test_optax_adam():
    opt_r = optax.adam(0.1)
    opt_z = zoptax.adam(0.1)
    params = {"w": np.array([1.0, 2.0])}
    state_r = opt_r.init(params)
    state_z = opt_z.init(params)

    grads = {"w": np.array([0.1, 0.2])}
    updates_r, new_state_r = opt_r.update(grads, state_r, params)
    updates_z, new_state_z = opt_z.update(grads, state_z, params)

    np.testing.assert_allclose(updates_z["w"], updates_r["w"], rtol=1e-5, atol=1e-5)


def test_optax_linear_schedule():
    sched_r = optax.linear_schedule(0.1, 0.01, 10)
    sched_z = zoptax.schedules.linear_schedule(0.1, 0.01, 10)

    np.testing.assert_allclose(sched_z(5), sched_r(5))


def test_optax_l2_loss():
    preds = np.array([1.0, 2.0])
    targets = np.array([1.5, 2.5])

    loss_r = optax.l2_loss(preds, targets)
    loss_z = zoptax.losses.l2_loss(preds, targets)

    np.testing.assert_allclose(loss_z, loss_r, rtol=1e-5, atol=1e-5)


def test_optax_softmax_cross_entropy():
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])

    loss_r = optax.softmax_cross_entropy(logits, labels)
    loss_z = zoptax.losses.softmax_cross_entropy(logits, labels)

    np.testing.assert_allclose(loss_z, loss_r, rtol=1e-5, atol=1e-5)
