"""Stochastic smoothing and differentiable perturbations (`zero_optax.perturbations`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, Union, runtime_checkable

from zero_optax._tree import (
    asarray,
    exp,
    reshape,
)
from zero_optax._types import (
    ArrayLike,
    DTypeLike,
    PyTree,
    Tensor,
)
from zero_optax.tree_utils._random import (
    _default_normal_sampler,
    _extract_seed_int,
    _split_seed_to_keys,
    tree_random_like,
)
from zero_optax.tree_utils._tree_math import (
    tree_add,
    tree_add_scale,
    tree_scalar_mul,
    tree_sub,
)


@runtime_checkable
class NoiseDistribution(Protocol):
    """Protocol for noise distributions used by `make_perturbed_fun`."""

    def sample(
        self,
        seed: ArrayLike | int,
        sample_shape: Sequence[int] = (),
        dtype: DTypeLike = "float32",
    ) -> Tensor:
        """Draw i.i.d. random samples of shape `sample_shape`.

        Args:
            seed: PRNG key array or integer seed.
            sample_shape: Shape of the output sample tensor.
            dtype: Data type of the samples.

        Returns:
            Sampled `Tensor` of shape `tuple(sample_shape)`.
        """
        ...

    def log_prob(self, inputs: ArrayLike) -> Tensor:
        """Compute element-wise log-probability density `log p(z)` at `inputs`.

        Args:
            inputs: Input tensor `z`.

        Returns:
            Element-wise log-probability Tensor.
        """
        ...


class Normal:
    """Standard normal distribution `N(0, 1)`."""

    def sample(
        self,
        seed: ArrayLike | int,
        sample_shape: Sequence[int] = (),
        dtype: DTypeLike = "float32",
    ) -> Tensor:
        """Draw standard normal samples of shape `sample_shape`.

        Args:
            seed: PRNG key array or integer seed.
            sample_shape: Shape of the output tensor.
            dtype: Data type of the samples.

        Returns:
            Standard normal `Tensor`.
        """
        key_t = asarray([0, _extract_seed_int(seed) & 0x7FFFFFFF], dtype="int32")
        return _default_normal_sampler(key_t, tuple(sample_shape), dtype=dtype)

    def log_prob(self, inputs: ArrayLike) -> Tensor:
        """Compute standard normal log-density `-0.5 * z^2` (up to additive constant).

        Args:
            inputs: Input tensor `z`.

        Returns:
            Element-wise log-probability Tensor `-0.5 * z^2`.
        """
        z = asarray(inputs)
        return -0.5 * (z * z)


class Gumbel:
    """Standard Gumbel distribution `Gumbel(0, 1)`."""

    def sample(
        self,
        seed: ArrayLike | int,
        sample_shape: Sequence[int] = (),
        dtype: DTypeLike = "float32",
    ) -> Tensor:
        """Draw standard Gumbel samples `-log(-log(U))` of shape `sample_shape`.

        Args:
            seed: PRNG key array or integer seed.
            sample_shape: Shape of the output tensor.
            dtype: Data type of the samples.

        Returns:
            Standard Gumbel `Tensor`.
        """
        shape_tup = tuple(sample_shape)
        total = 1
        for dim in shape_tup:
            total *= dim
        state = (_extract_seed_int(seed) ^ 0x5A5A5A5A) & 0xFFFFFFFF
        vals: list[float] = []
        for idx in range(total):
            state = (1664525 * state + 1013904223 + idx) & 0xFFFFFFFF
            u = min(max((state + 1.0) / 4294967297.0, 1e-7), 1.0 - 1e-7)
            vals.append(-math.log(-math.log(u)))
        if total == 0:
            return reshape(asarray([], dtype=dtype), shape_tup)
        if shape_tup == ():
            return asarray(vals[0], dtype=dtype)
        return reshape(asarray(vals, dtype=dtype), shape_tup)

    def log_prob(self, inputs: ArrayLike) -> Tensor:
        """Compute standard Gumbel log-density `-z - exp(-z)`.

        Args:
            inputs: Input tensor `z`.

        Returns:
            Element-wise log-probability Tensor `-z - exp(-z)`.
        """
        z = asarray(inputs)
        return -z - exp(-z)


_DEFAULT_GUMBEL = Gumbel()

PerturbedFn = Callable[[Union[ArrayLike, int], PyTree], PyTree]


def make_perturbed_fun(
    fun: Callable[[PyTree], PyTree],
    num_samples: int = 1000,
    sigma: float = 0.1,
    noise: NoiseDistribution = _DEFAULT_GUMBEL,
    use_baseline: bool = True,
) -> PerturbedFn:
    """Return a differentiable Monte-Carlo perturbation approximation of `fun`.

    Args:
        fun: Function mapping a PyTree of tensors to a PyTree of tensors.
        num_samples: Number of random perturbations to average over (`>= 1`).
        sigma: Scale standard deviation of the additive noise `x + sigma * z`.
        noise: Noise distribution implementing `sample` and `log_prob` (default `Gumbel()`).
        use_baseline: Whether to use `fun(inputs)` as a control-variate baseline.

    Returns:
        Callable `perturbed_fun(rng, inputs)` evaluating the smoothed expectation
        `E_{z}[fun(inputs + sigma * z)]`.
    """

    def _sampler_adapter(
        key: Tensor,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> Tensor:
        """Adapter passing `(key, shape, dtype)` to `noise.sample`.

        Args:
            key: PRNG key Tensor.
            shape: Target tensor shape.
            dtype: Target dtype.

        Returns:
            Sampled noise Tensor.
        """
        return noise.sample(key, sample_shape=shape, dtype=dtype)

    def _perturbed_fun(
        rng: ArrayLike | int,
        inputs: PyTree,
    ) -> PyTree:
        """Evaluate `E_z[fun(inputs + sigma * z)]` over `num_samples` Monte-Carlo draws.

        Args:
            rng: PRNG key array or integer seed.
            inputs: Input PyTree to perturb and evaluate.

        Returns:
            Averaged output PyTree over `num_samples` perturbations.
        """
        n_samp = max(int(num_samples), 1)
        base_out = fun(inputs) if use_baseline else None
        subkeys = _split_seed_to_keys(_extract_seed_int(rng), n_samp)

        acc_out: PyTree = None
        for idx, sk in enumerate(subkeys):
            z_tree = tree_random_like(sk, inputs, sampler=_sampler_adapter)
            pert_in = tree_add_scale(inputs, float(sigma), z_tree)
            y_i = fun(pert_in)
            acc_out = y_i if idx == 0 else tree_add(acc_out, y_i)

        mean_out = tree_scalar_mul(1.0 / float(n_samp), acc_out)
        if base_out is not None:
            diff = tree_sub(mean_out, base_out)
            return tree_add(base_out, diff)
        return mean_out

    return _perturbed_fun


__all__ = [
    "Gumbel",
    "NoiseDistribution",
    "Normal",
    "PerturbedFn",
    "make_perturbed_fun",
]
