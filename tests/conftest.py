"""Shared pytest fixtures. Makes the project root importable and provides small datasets."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import MnistData, load_mnist  # noqa: E402


@pytest.fixture(scope="session")
def mnist() -> MnistData:
    """The real dataset (from the local cache shipped in data/)."""
    return load_mnist(allow_download=True)


@pytest.fixture(scope="session")
def small_mnist(mnist) -> MnistData:
    """A small, fast slice of real MNIST for model tests."""
    return MnistData(mnist.x_train[:3000], mnist.y_train[:3000], mnist.x_test[:1000], mnist.y_test[:1000], "slice")


@pytest.fixture
def synthetic_digit() -> np.ndarray:
    """A 200x200 white canvas with a thick black vertical bar (looks like a '1')."""
    img = np.full((200, 200), 255, dtype=np.uint8)
    img[40:170, 90:115] = 0
    return img
