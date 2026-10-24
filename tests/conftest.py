"""Pytest configuration and eager execution fixtures for `zero_optax`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from ml_switcheroo_compiler.core import EagerMode


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Enable `ml-switcheroo-compiler` EagerMode across every test by default.

    Yields:
        None while inside the `EagerMode` context manager.
    """
    with EagerMode():
        yield
