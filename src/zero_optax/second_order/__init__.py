"""Second-order curvature utilities (`hvp`, `hessian_diag`, `fisher_diag`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from zero_optax._tree import asarray
from zero_optax._types import (
    ArrayLike,
    Params,
    Scalar,
    Tensor,
)
from zero_optax.tree_utils._tree_math import _flatten_tensor_floats
from zero_optax.wrappers import _flatten_pytree_to_1d, _unflatten_1d_to_pytree

LossFn = Callable[[Params, ArrayLike, ArrayLike], Union[Scalar, Tensor]]


def hvp(
    loss: LossFn,
    v: ArrayLike,
    params: Params,
    inputs: ArrayLike,
    targets: ArrayLike,
) -> Tensor:
    """Perform a Hessian-vector product `H(loss)(params) @ v` flattened to 1D.

    Args:
        loss: Scalar loss function `loss(params, inputs, targets)`.
        v: 1D vector of length equal to the total number of parameters in `params`.
        params: Model parameters PyTree.
        inputs: Input batch passed to `loss`.
        targets: Target batch passed to `loss`.

    Returns:
        1D `Tensor` representing `H(params) @ v`.

    Raises:
        ValueError: If `v` is not a 1D vector matching the flattened size of `params`.
    """
    flat_p_t, meta = _flatten_pytree_to_1d(params)
    p_vec = _flatten_tensor_floats(flat_p_t)
    v_t = asarray(v)
    v_vec = _flatten_tensor_floats(v_t)
    n = len(p_vec)
    if v_t.ndim != 1 or len(v_vec) != n:
        raise ValueError(f"`v` must be a 1D vector of size {n}, got shape {v_t.shape}.")

    def _eval_flat(vec: list[float]) -> float:
        """Evaluate `loss` at flattened parameter vector `vec`.

        Args:
            vec: Flat parameter list.

        Returns:
            Scalar float loss value.
        """
        rebuilt = _unflatten_1d_to_pytree(asarray(vec), params, meta)
        return float(asarray(loss(rebuilt, inputs, targets)).item())

    def _grad_flat(vec: list[float], eps_g: float = 1e-4) -> list[float]:
        """Compute central-difference gradient of `_eval_flat` at `vec`.

        Args:
            vec: Flat parameter list.
            eps_g: Finite-difference step size.

        Returns:
            Flat gradient list of length `n`.
        """
        g_out: list[float] = []
        for idx in range(n):
            vp = list(vec)
            vm = list(vec)
            vp[idx] += eps_g
            vm[idx] -= eps_g
            g_out.append((_eval_flat(vp) - _eval_flat(vm)) / (2.0 * eps_g))
        return g_out

    eps_h = 1e-3
    p_plus = [p_vec[i] + eps_h * v_vec[i] for i in range(n)]
    p_minus = [p_vec[i] - eps_h * v_vec[i] for i in range(n)]
    g_plus = _grad_flat(p_plus)
    g_minus = _grad_flat(p_minus)
    hvp_vals = [(g_plus[i] - g_minus[i]) / (2.0 * eps_h) for i in range(n)]
    return asarray(hvp_vals)


def hessian_diag(
    loss: LossFn,
    params: Params,
    inputs: ArrayLike,
    targets: ArrayLike,
) -> Tensor:
    """Compute the diagonal of the Hessian of `loss` at `(params, inputs, targets)`.

    Args:
        loss: Scalar loss function `loss(params, inputs, targets)`.
        params: Model parameters PyTree.
        inputs: Input batch passed to `loss`.
        targets: Target batch passed to `loss`.

    Returns:
        1D `Tensor` containing the diagonal entries of the Hessian matrix.
    """
    flat_p_t, meta = _flatten_pytree_to_1d(params)
    p_vec = _flatten_tensor_floats(flat_p_t)
    n = len(p_vec)

    def _eval_flat(vec: list[float]) -> float:
        """Evaluate `loss` at flattened parameter vector `vec`.

        Args:
            vec: Flat parameter list.

        Returns:
            Scalar float loss value.
        """
        rebuilt = _unflatten_1d_to_pytree(asarray(vec), params, meta)
        return float(asarray(loss(rebuilt, inputs, targets)).item())

    f0 = _eval_flat(p_vec)
    eps = 1e-3
    diag: list[float] = []
    for idx in range(n):
        vp = list(p_vec)
        vm = list(p_vec)
        vp[idx] += eps
        vm[idx] -= eps
        fp = _eval_flat(vp)
        fm = _eval_flat(vm)
        diag.append((fp - 2.0 * f0 + fm) / (eps * eps))
    return asarray(diag)


def fisher_diag(
    negative_log_likelihood: LossFn,
    params: Params,
    inputs: ArrayLike,
    targets: ArrayLike,
) -> Tensor:
    """Compute the diagonal of the observed Fisher information matrix `(grad(nll) ** 2)`.

    Args:
        negative_log_likelihood: Negative log-likelihood loss callable `fn(params, inputs, targets)`.
        params: Model parameters PyTree.
        inputs: Input batch passed to `negative_log_likelihood`.
        targets: Target batch passed to `negative_log_likelihood`.

    Returns:
        1D `Tensor` containing the squared gradient entries.
    """
    flat_p_t, meta = _flatten_pytree_to_1d(params)
    p_vec = _flatten_tensor_floats(flat_p_t)
    n = len(p_vec)

    def _eval_flat(vec: list[float]) -> float:
        """Evaluate `negative_log_likelihood` at flattened parameter vector `vec`.

        Args:
            vec: Flat parameter list.

        Returns:
            Scalar float NLL value.
        """
        rebuilt = _unflatten_1d_to_pytree(asarray(vec), params, meta)
        return float(asarray(negative_log_likelihood(rebuilt, inputs, targets)).item())

    eps = 1e-4
    f_diag: list[float] = []
    for idx in range(n):
        vp = list(p_vec)
        vm = list(p_vec)
        vp[idx] += eps
        vm[idx] -= eps
        g_i = (_eval_flat(vp) - _eval_flat(vm)) / (2.0 * eps)
        f_diag.append(g_i * g_i)
    return asarray(f_diag)


__all__ = [
    "LossFn",
    "fisher_diag",
    "hessian_diag",
    "hvp",
]
