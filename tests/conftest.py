import numpy as np
import pytest


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    return rng.normal(size=(200, 2))


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 2))
