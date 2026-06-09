import numpy as np


def assert_allclose_optax(actual, expected, rtol=1e-5, atol=1e-5):
    """Assert allclose between actual and expected arrays."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_gradients_allclose(grad_fn, args, expected_grad, rtol=1e-5, atol=1e-5):
    """Assert gradients are allclose."""
    actual_grad = grad_fn(*args)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=rtol, atol=atol)


def generate_test_inputs(shape, dtype=np.float32):
    """Generate deterministic random test inputs."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(shape, dtype=dtype)


def numeric_grad(fun, x, eps=1e-5):
    """Compute numerical gradient using central difference."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        x_minus = x.copy()
        x_minus.flat[i] -= eps
        grad.flat[i] = (fun(x_plus) - fun(x_minus)) / (2 * eps)
    return grad
