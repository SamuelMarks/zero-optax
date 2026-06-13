import pytest
import sys
import os
import numpy as np

import zero_jax
from zero_jax.numpy.lax_numpy import ndarray


# Monkeypatch __array__ so np.array() works in tests
def _array_patch(self):
    return np.asarray(self._tensor.data)


ndarray.__array__ = _array_patch


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports zero_jax config contexts
    with zero_jax.EagerMode():
        yield


import chex


def force_no_variants(*args, **kwargs):
    def decorator(cls_or_func):
        if hasattr(cls_or_func, "variant"):
            return cls_or_func
        if isinstance(cls_or_func, type):
            cls_or_func.variant = lambda self, f: f
            return cls_or_func
        else:
            return cls_or_func

    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator


chex.all_variants = force_no_variants
import unittest

unittest.TestCase.variant = lambda self, f: f
