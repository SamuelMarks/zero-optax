import numpy as np


def numeric_grad(fn, x, eps=1e-4):
    x_np = np.array(x)
    grad = np.zeros_like(x_np)
    flat_x = x_np.flatten()
    flat_grad = np.zeros_like(flat_x)
    for i in range(len(flat_x)):
        orig = flat_x[i]
        flat_x[i] = orig + eps
        val_plus = float(np.array(fn(flat_x.reshape(x_np.shape))))
        flat_x[i] = orig - eps
        val_minus = float(np.array(fn(flat_x.reshape(x_np.shape))))
        flat_x[i] = orig
        flat_grad[i] = (val_plus - val_minus) / (2 * eps)
    return flat_grad.reshape(x_np.shape)


def assert_allclose_optax(a, b, **kwargs):
    if hasattr(a, "numpy"):
        a = a.numpy()
    if hasattr(b, "numpy"):
        b = b.numpy()
    np.testing.assert_allclose(a, b, **kwargs)
