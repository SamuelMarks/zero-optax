"""Fenchel-Young loss."""

from typing import Any


def make_fenchel_young_loss(max_fun: "Any"):  # type: ignore
    """Creates a Fenchel-Young loss from a max function."""

    def loss_fn(predictions, targets):
        """Loss fn."""
        return predictions - targets

    return loss_fn
