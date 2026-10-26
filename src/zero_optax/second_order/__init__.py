"""Second-order curvature utilities (`hvp`, `hessian_diag`, `fisher_diag`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from zero_optax._tree import arange, asarray, where
from zero_optax._types import (
    ArrayLike,
    Params,
    Scalar,
    Tensor,
)
from zero_optax.wrappers import _flatten_pytree_to_1d, _unflatten_1d_to_pytree

LossFn = Callable[[Params, ArrayLike, ArrayLike], Union[Scalar, Tensor]]


def _flat_grad_1d(
    loss: LossFn,
    flat_p: Tensor,
    params_template: Params,
    meta: list[tuple[tuple[int, ...], str] | None],
    inputs: ArrayLike,
    targets: ArrayLike,
    eps: float = 1e-4,
) -> Tensor:
    """Compute central-difference gradient of `loss` with respect to 1D `flat_p`.

    Args:
        loss: Scalar loss function `loss(params, inputs, targets)`.
        flat_p: 1D flattened parameter `Tensor`.
        params_template: Template PyTree structure for `params`.
        meta: Leaf shape and dtype metadata from `_flatten_pytree_to_1d`.
        inputs: Input batch passed to `loss`.
        targets: Target batch passed to `loss`.
        eps: Finite-difference step size.

    Returns:
        1D gradient `Tensor` of the same shape as `flat_p`.
    """
    n = flat_p.shape[0]
    idx_vec = arange(n)
    g_out: list[float] = []
    for idx in range(n):
        basis = where(idx_vec == idx, eps, 0.0)
        p_plus = _unflatten_1d_to_pytree(flat_p + basis, params_template, meta)
        p_minus = _unflatten_1d_to_pytree(flat_p - basis, params_template, meta)
        fp = float(asarray(loss(p_plus, inputs, targets)).item())
        fm = float(asarray(loss(p_minus, inputs, targets)).item())
        g_out.append((fp - fm) / (2.0 * eps))
    return asarray(g_out)


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
    v_t = asarray(v)
    n = flat_p_t.shape[0]
    if v_t.ndim != 1 or v_t.shape[0] != n:
        raise ValueError(f"`v` must be a 1D vector of size {n}, got shape {v_t.shape}.")

    eps_h = 1e-3
    g_plus = _flat_grad_1d(loss, flat_p_t + eps_h * v_t, params, meta, inputs, targets)
    g_minus = _flat_grad_1d(loss, flat_p_t - eps_h * v_t, params, meta, inputs, targets)
    return (g_plus - g_minus) / (2.0 * eps_h)


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
    n = flat_p_t.shape[0]
    f0 = float(asarray(loss(params, inputs, targets)).item())
    eps = 1e-3
    idx_vec = arange(n)
    diag: list[float] = []
    for idx in range(n):
        basis = where(idx_vec == idx, eps, 0.0)
        p_plus = _unflatten_1d_to_pytree(flat_p_t + basis, params, meta)
        p_minus = _unflatten_1d_to_pytree(flat_p_t - basis, params, meta)
        fp = float(asarray(loss(p_plus, inputs, targets)).item())
        fm = float(asarray(loss(p_minus, inputs, targets)).item())
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
    g_t = _flat_grad_1d(
        negative_log_likelihood, flat_p_t, params, meta, inputs, targets
    )
    return g_t * g_t


__all__ = [
    "LossFn",
    "fisher_diag",
    "hessian_diag",
    "hvp",
]
