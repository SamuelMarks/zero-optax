import numpy as np


def assert_allclose_optax(actual, expected, rtol=1e-5, atol=1e-5):
    """Verify numeric equivalence matches optax expectations."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def numeric_grad(fn, x, eps=1e-4):
    """Compute numerical gradient for testing."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = np.array(x, copy=True)
        x_plus.flat[i] += eps
        x_minus = np.array(x, copy=True)
        x_minus.flat[i] -= eps
        grad.flat[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad
