import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo


@pytest.fixture(autouse=True)
def switcheroo_config():
    # Unified pytest configuration that imports switcheroo config contexts
    with ml_switcheroo.EagerMode():
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
