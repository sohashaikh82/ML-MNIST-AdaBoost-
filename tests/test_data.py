"""Dataset loading and validation tests."""

import numpy as np
import pytest

from src.data_loader import DatasetUnavailableError, MnistData, load_mnist, validate_dataset


def test_official_split_and_shapes(mnist):
    assert mnist.x_train.shape == (60000, 28, 28)
    assert mnist.x_test.shape == (10000, 28, 28)
    assert mnist.y_train.shape == (60000,)
    assert mnist.y_test.shape == (10000,)
    assert mnist.x_train.dtype == np.uint8


def test_labels_are_digits(mnist):
    assert set(np.unique(mnist.y_train)) == set(range(10))
    assert set(np.unique(mnist.y_test)) == set(range(10))


def test_validation_report_measures_quality(mnist):
    rep = validate_dataset(mnist)
    assert rep["problems"] == []
    assert rep["labels_ok"]
    assert rep["pixel_min"] >= 0 and rep["pixel_max"] <= 255
    assert rep["missing_train_pixels"] == 0 and rep["missing_test_pixels"] == 0
    assert sum(rep["train_class_counts"]) == 60000
    assert rep["train_test_overlap"] >= 0


def test_missing_source_raises_helpful_error(tmp_path):
    with pytest.raises(DatasetUnavailableError, match="mnist.npz"):
        load_mnist(tmp_path / "nope.npz", allow_download=False)


def test_corrupt_cache_is_rejected(tmp_path):
    bad = tmp_path / "mnist.npz"
    bad.write_bytes(b"not a real npz")
    with pytest.raises(DatasetUnavailableError):
        load_mnist(bad, allow_download=False)


def test_validation_flags_bad_shapes():
    data = MnistData(np.zeros((5, 27, 28), np.uint8), np.zeros(5, int), np.zeros((2, 28, 28), np.uint8),
                     np.zeros(3, int), "fake")
    rep = validate_dataset(data)
    assert any("x_train" in p for p in rep["problems"])
    assert any("y_test" in p for p in rep["problems"])
