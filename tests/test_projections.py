"""Comprehensive unit and reference tests for `zero_optax.projections`."""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax import projections, tree_utils
from zero_optax._tree import asarray
from zero_optax._types import PyTree
from zero_optax.projections._projections import _project_unit_simplex_1d


def test_projections_box_non_negative_hypercube_linf() -> None:
    """Test projection_non_negative, projection_box, projection_hypercube, and projection_linf_ball."""
    t: dict[str, PyTree] = {"w": asarray([-2.0, 0.5, 3.0]), "n": None}

    nn = projections.projection_non_negative(t)
    assert_allclose_optax(get_tree_item(nn, "w"), np.array([0.0, 0.5, 3.0]))

    box_scalar = projections.projection_box(t, lower=-1.0, upper=2.0)
    assert_allclose_optax(get_tree_item(box_scalar, "w"), np.array([-1.0, 0.5, 2.0]))

    box_none = projections.projection_box(t, lower=None, upper=None)
    assert_allclose_optax(get_tree_item(box_none, "w"), np.array([-2.0, 0.5, 3.0]))

    box_tree = projections.projection_box(
        t,
        lower={"w": asarray([-0.5, 0.0, 0.0]), "n": None},
        upper={"w": asarray([1.0, 1.0, 1.5]), "n": None},
    )
    assert_allclose_optax(get_tree_item(box_tree, "w"), np.array([-0.5, 0.5, 1.5]))

    hc = projections.projection_hypercube(t, scale=2.0)
    assert_allclose_optax(get_tree_item(hc, "w"), np.array([0.0, 0.5, 2.0]))

    linf = projections.projection_linf_ball(t, scale=1.5)
    assert_allclose_optax(get_tree_item(linf, "w"), np.array([-1.5, 0.5, 1.5]))


def test_projections_l2_l1_and_simplex() -> None:
    """Test L2 sphere/ball, simplex, and L1 sphere/ball projections."""
    t: dict[str, PyTree] = {
        "w": asarray([3.0, -4.0]),
        "s": asarray(0.0),
        "n": None,
    }

    l2_sph = projections.projection_l2_sphere(t, scale=2.5)
    assert float(tree_utils.tree_l2_norm(l2_sph).item()) == pytest.approx(2.5)

    l2_ball_inside = projections.projection_l2_ball(t, scale=10.0)
    assert l2_ball_inside is t
    l2_ball_outside = projections.projection_l2_ball(t, scale=2.5)
    assert float(tree_utils.tree_l2_norm(l2_ball_outside).item()) == pytest.approx(2.5)

    assert _project_unit_simplex_1d([]) == []
    with pytest.raises(ValueError, match="scale must be non-negative"):
        projections.projection_simplex(t, scale=-1.0)
    simp_zero = projections.projection_simplex(t, scale=0.0)
    assert float(tree_utils.tree_sum(simp_zero).item()) == pytest.approx(0.0)

    simp_none = projections.projection_simplex(
        {"w": asarray([2.5, 3.2]), "b": asarray(0.5), "n": None}, scale=None
    )
    assert float(tree_utils.tree_sum(simp_none).item()) == pytest.approx(1.0, rel=1e-5)

    l1_sph = projections.projection_l1_sphere(t, scale=2.0)
    assert float(tree_utils.tree_l1_norm(l1_sph).item()) == pytest.approx(2.0, rel=1e-5)

    l1_ball_in = projections.projection_l1_ball(t, scale=10.0)
    assert l1_ball_in is t
    l1_ball_out = projections.projection_l1_ball(t, scale=2.0)
    assert float(tree_utils.tree_l1_norm(l1_ball_out).item()) == pytest.approx(
        2.0, rel=1e-5
    )


def test_projections_vector_hyperplane_halfspace() -> None:
    """Test projection_vector, projection_hyperplane, and projection_halfspace."""
    x: dict[str, PyTree] = {"w": asarray([3.0, 4.0])}
    a: dict[str, PyTree] = {"w": asarray([1.0, 0.0])}

    pv = projections.projection_vector(x, a)
    assert_allclose_optax(get_tree_item(pv, "w"), np.array([3.0, 0.0]))

    php = projections.projection_hyperplane(x, a, 1.5)
    assert_allclose_optax(get_tree_item(php, "w"), np.array([1.5, 4.0]))

    phs_out = projections.projection_halfspace(x, a, 1.5)
    assert_allclose_optax(get_tree_item(phs_out, "w"), np.array([1.5, 4.0]))

    phs_in = projections.projection_halfspace(x, a, 5.0)
    assert_allclose_optax(get_tree_item(phs_in, "w"), np.array([3.0, 4.0]))
