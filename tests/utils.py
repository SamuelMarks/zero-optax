"""Numerical gradient and assertion helpers for `zero_optax` tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
import numpy.typing as npt
from ml_switcheroo_compiler.core.tensor import Tensor

from zero_optax._tree import asarray
from zero_optax._types import ArrayLike, PyTree

FloatArray = npt.NDArray[np.float64]
GenericArray = npt.NDArray[Union[np.float64, np.float32, np.int64, np.int32, np.bool_]]


def get_tree_item(tree: PyTree, key: str) -> Tensor:
    """Extract a `Tensor` leaf from a dictionary `PyTree` by `key`.

    Args:
        tree: Dictionary `PyTree` containing parameter or update leaves.
        key: Key string to look up.

    Returns:
        Extracted `Tensor` leaf.

    Raises:
        TypeError: If `tree` is not a mapping or the value at `key` is not a `Tensor`.
    """
    if not isinstance(tree, dict):
        raise TypeError(f"Expected dict PyTree, got {type(tree)!r}.")
    val = tree[key]
    if not isinstance(val, Tensor):
        raise TypeError(f"Expected Tensor at key {key!r}, got {type(val)!r}.")
    return val


def to_numpy(x: ArrayLike | GenericArray) -> FloatArray:
    """Convert a `Tensor`, sequence, scalar, or NumPy array into a `float64` NumPy array.

    Args:
        x: Input tensor or array-like value.

    Returns:
        NumPy `float64` `ndarray` representation.
    """
    if isinstance(x, Tensor):
        return np.asarray(x.numpy(), dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    return np.asarray(asarray(x).numpy(), dtype=np.float64)


def numeric_grad(
    fn: Callable[[Tensor], Tensor],
    x: ArrayLike | GenericArray,
    eps: float = 1e-4,
) -> FloatArray:
    """Compute central finite-difference numerical gradients of scalar function `fn`.

    Args:
        fn: Scalar-valued loss callable taking a `Tensor`.
        x: Point at which to evaluate the gradient.
        eps: Finite-difference step perturbation size.

    Returns:
        NumPy array of numerical gradients with the same shape as `x`.
    """
    x_np = to_numpy(x)
    flat_x = x_np.flatten()
    flat_grad = np.zeros_like(flat_x)
    for idx in range(len(flat_x)):
        orig = float(flat_x[idx])
        flat_x[idx] = orig + eps
        pos_val = float(
            to_numpy(fn(asarray(flat_x.reshape(x_np.shape).tolist()))).item()
        )
        flat_x[idx] = orig - eps
        neg_val = float(
            to_numpy(fn(asarray(flat_x.reshape(x_np.shape).tolist()))).item()
        )
        flat_x[idx] = orig
        flat_grad[idx] = (pos_val - neg_val) / (2.0 * eps)
    return flat_grad.reshape(x_np.shape)


def assert_allclose_optax(
    a: ArrayLike | GenericArray,
    b: ArrayLike | GenericArray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert that two tensors or array-likes are numerically close within tolerances.

    Args:
        a: Actual tensor or array-like output.
        b: Expected tensor or array-like output.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    np.testing.assert_allclose(to_numpy(a), to_numpy(b), rtol=rtol, atol=atol)
