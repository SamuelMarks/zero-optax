"""Fenchel-Young convex conjugate loss factory."""

from __future__ import annotations

from typing import Protocol

from zero_optax._tree import asarray, reduce_sum, reshape, stack
from zero_optax._types import ArrayLike, Tensor


class MaxFun(Protocol):
    """Protocol for a regularized maximum callable used in `make_fenchel_young_loss`."""

    def __call__(
        self,
        scores: Tensor,
        /,
        *args: float | int | str | bool,
        **kwargs: float | int | str | bool,
    ) -> Tensor:
        """Compute the regularized maximum of `scores`.

        Args:
            scores: Unnormalized score Tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Regularized maximum Tensor.
        """
        ...


class FenchelYoungLossFn(Protocol):
    """Protocol for the loss callable returned by `make_fenchel_young_loss`."""

    def __call__(
        self,
        scores: ArrayLike,
        targets: ArrayLike,
        *args: float | int | str | bool,
        **kwargs: float | int | str | bool,
    ) -> Tensor:
        """Evaluate the Fenchel-Young loss.

        Args:
            scores: Unnormalized model scores.
            targets: Target probabilities.
            *args: Additional positional arguments forwarded to `max_fun`.
            **kwargs: Additional keyword arguments forwarded to `max_fun`.

        Returns:
            Fenchel-Young loss Tensor.
        """
        ...


def make_fenchel_young_loss(
    max_fun: MaxFun,
) -> FenchelYoungLossFn:
    """Construct a Fenchel-Young loss function from a regularized maximum function `max_fun`.

    Given a convex regularizer `Omega` with Fenchel conjugate `max_fun(scores) = Omega*(scores)`,
    the Fenchel-Young loss is `max_fun(scores) - <targets, scores>` along the last axis.

    Args:
        max_fun: Callable computing the regularized maximum over the last dimension
            (or a 1D vector).

    Returns:
        A callable `fenchel_young_loss(scores, targets, *args, **kwargs) -> Tensor`.
    """

    def fenchel_young_loss(
        scores: ArrayLike,
        targets: ArrayLike,
        *args: float | int | str | bool,
        **kwargs: float | int | str | bool,
    ) -> Tensor:
        """Evaluate the Fenchel-Young loss `max_fun(scores) - sum(targets * scores, axis=-1)`.

        Args:
            scores: Unnormalized model scores of shape `[..., num_classes]`.
            targets: Target probability distribution or ground-truth vector of shape
                matching `scores`.
            *args: Additional positional arguments forwarded to `max_fun`.
            **kwargs: Additional keyword arguments forwarded to `max_fun`.

        Returns:
            Fenchel-Young loss Tensor of shape `scores.shape[:-1]`.
        """
        s_arr = asarray(scores)
        t_arr = asarray(targets)
        raw_max = max_fun(s_arr, *args, **kwargs)
        max_val = asarray(raw_max)
        inner_prod = reduce_sum(t_arr * s_arr, axis=-1)
        if max_val.shape != inner_prod.shape and s_arr.ndim > 1:
            lead_shape = s_arr.shape[:-1]
            num_classes = s_arr.shape[-1]
            num_examples = 1
            for dim in lead_shape:
                num_examples *= dim
            s_2d = reshape(s_arr, (num_examples, num_classes))
            row_maxes = [
                asarray(max_fun(s_2d[i], *args, **kwargs)) for i in range(num_examples)
            ]
            max_val = reshape(stack(row_maxes, axis=0), lead_shape)
        return max_val - inner_prod

    return fenchel_young_loss


__all__ = [
    "MaxFun",
    "make_fenchel_young_loss",
]
