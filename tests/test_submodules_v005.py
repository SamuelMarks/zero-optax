"""Comprehensive unit and reference tests for `second_order`, `assignment`, and `perturbations`."""

from __future__ import annotations

import numpy as np
import pytest

import zero_optax
from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax._tree import asarray, reduce_sum
from zero_optax._types import ArrayLike, Params, PyTree, Tensor


def test_second_order_hvp_hessian_diag_fisher_diag() -> None:
    """Test hvp, hessian_diag, and fisher_diag on quadratic objective."""
    params: dict[str, PyTree] = {"w": asarray([2.0, -3.0])}
    inputs = asarray([1.0, 1.0])
    targets = asarray([0.0, 0.0])

    def _loss(p: Params, x: ArrayLike, y: ArrayLike) -> Tensor:
        """Compute `0.5 * sum((w * x - y) ** 2)`.

        Args:
            p: Parameters dict.
            x: Inputs tensor.
            y: Targets tensor.

        Returns:
            Scalar loss Tensor.
        """
        if not isinstance(p, dict):
            return asarray(0.0)
        w = asarray(p["w"])
        diff = w * asarray(x) - asarray(y)
        return 0.5 * reduce_sum(diff * diff)

    v = asarray([1.0, 2.0])
    hv = zero_optax.second_order.hvp(_loss, v, params, inputs, targets)
    assert_allclose_optax(hv, np.array([1.0, 2.0]), rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="1D vector"):
        zero_optax.second_order.hvp(_loss, asarray([1.0]), params, inputs, targets)

    hd = zero_optax.second_order.hessian_diag(_loss, params, inputs, targets)
    assert_allclose_optax(hd, np.array([1.0, 1.0]), rtol=1e-2, atol=1e-2)

    fd = zero_optax.second_order.fisher_diag(_loss, params, inputs, targets)
    assert_allclose_optax(fd, np.array([4.0, 9.0]), rtol=1e-3, atol=1e-3)


def test_assignment_hungarian_algorithm() -> None:
    """Test base_hungarian_algorithm and hungarian_algorithm on tall, wide, square, and empty matrices."""
    from zero_optax.assignment import _solve_hungarian_rect

    assert _solve_hungarian_rect([], 0, 0) == ([], [])
    with pytest.raises(ValueError, match="2D"):
        zero_optax.assignment.base_hungarian_algorithm(asarray([1.0, 2.0]))

    r_empty, c_empty = zero_optax.assignment.hungarian_algorithm(
        zero_optax._tree.zeros((0, 3))
    )
    assert r_empty.shape == (0,) and c_empty.shape == (0,)

    cost_tall = asarray(
        [
            [8.0, 4.0, 7.0],
            [5.0, 2.0, 3.0],
            [9.0, 6.0, 7.0],
            [9.0, 4.0, 8.0],
        ]
    )
    ri, ci = zero_optax.assignment.hungarian_algorithm(cost_tall)
    assert_allclose_optax(ri, np.array([0, 1, 3]))
    assert_allclose_optax(ci, np.array([0, 2, 1]))

    cost_wide = asarray(
        [
            [8.0, 5.0, 9.0, 9.0],
            [4.0, 2.0, 6.0, 4.0],
            [7.0, 3.0, 7.0, 8.0],
        ]
    )
    ri_w, ci_w = zero_optax.assignment.base_hungarian_algorithm(cost_wide)
    assert_allclose_optax(ri_w, np.array([0, 1, 2]))
    assert_allclose_optax(ci_w, np.array([0, 3, 1]))


def test_perturbations_normal_gumbel_and_make_perturbed_fun() -> None:
    """Test Normal, Gumbel, and make_perturbed_fun."""
    norm = zero_optax.perturbations.Normal()
    s_n = norm.sample(0, (3,))
    assert s_n.shape == (3,)
    lp_n = norm.log_prob(asarray([0.0, 1.0]))
    assert_allclose_optax(lp_n, np.array([0.0, -0.5]))

    gumb = zero_optax.perturbations.Gumbel()
    assert gumb.sample(0, ()).shape == ()
    assert gumb.sample(0, (0,)).shape == (0,)
    assert gumb.sample(0, (2, 2)).shape == (2, 2)
    lp_g = gumb.log_prob(asarray([0.0]))
    assert_allclose_optax(lp_g, np.array([-1.0]))

    def _square_tree(x: PyTree) -> PyTree:
        """Square leaf `'w'` in dict PyTree.

        Args:
            x: Input dict PyTree.

        Returns:
            Squared dict PyTree.
        """
        if not isinstance(x, dict):
            return x
        w = asarray(x["w"])
        return {"w": w * w}

    p_fn_base = zero_optax.perturbations.make_perturbed_fun(
        _square_tree, num_samples=10, sigma=0.01, noise=norm, use_baseline=True
    )
    out_base = p_fn_base(42, {"w": asarray([1.0, 2.0])})
    assert_allclose_optax(
        get_tree_item(out_base, "w"), np.array([1.0, 4.0]), rtol=1e-2, atol=1e-2
    )

    p_fn_nobase = zero_optax.perturbations.make_perturbed_fun(
        _square_tree, num_samples=5, sigma=0.01, noise=gumb, use_baseline=False
    )
    out_nobase = p_fn_nobase(42, {"w": asarray([1.0, 2.0])})
    assert_allclose_optax(
        get_tree_item(out_nobase, "w"), np.array([1.0, 4.0]), rtol=5e-2, atol=5e-2
    )
