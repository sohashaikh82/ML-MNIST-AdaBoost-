"""Preprocessing tests: normalisation, flattening, subsampling and user-image pipeline."""

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    BlankImageError,
    InvalidImageError,
    flatten_images,
    normalize_pixels,
    prepare_features,
    preprocess_user_image,
    split_if_needed,
    stratified_subsample,
)


def test_normalize_range_and_dtype():
    x = np.array([[0, 128, 255]], dtype=np.uint8)
    out = normalize_pixels(x)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.isclose(out[0, 1], 128 / 255)


def test_normalize_rejects_out_of_range():
    with pytest.raises(ValueError):
        normalize_pixels(np.array([300.0]))


def test_flatten_dimensions():
    assert flatten_images(np.zeros((5, 28, 28))).shape == (5, 784)
    assert flatten_images(np.zeros((28, 28))).shape == (1, 784)
    with pytest.raises(ValueError):
        flatten_images(np.zeros((5, 20, 20)))


def test_prepare_features_matches_manual(mnist):
    X = prepare_features(mnist.x_test[:3])
    assert X.shape == (3, 784)
    assert np.allclose(X[0], mnist.x_test[0].reshape(-1) / 255.0)


def test_stratified_subsample_is_balanced_and_reproducible(mnist):
    X = np.zeros((len(mnist.y_train), 1))
    _, y1, i1 = stratified_subsample(X, mnist.y_train, 5000, random_state=1)
    _, y2, i2 = stratified_subsample(X, mnist.y_train, 5000, random_state=1)
    assert len(y1) == 5000 and np.array_equal(i1, i2)
    full = np.bincount(mnist.y_train) / len(mnist.y_train)
    sub = np.bincount(y1, minlength=10) / 5000
    assert np.abs(full - sub).max() < 0.005


def test_subsample_full_size_returns_everything():
    X, y = np.arange(20).reshape(10, 2), np.array([0, 1] * 5)
    Xs, ys, idx = stratified_subsample(X, y, 50)
    assert len(ys) == 10 and np.array_equal(idx, np.arange(10))
    with pytest.raises(ValueError):
        stratified_subsample(X, y, 0)


def test_split_if_needed_keeps_classes():
    X, y = np.random.rand(70, 3), np.repeat(np.arange(7), 10)
    Xtr, Xte, ytr, yte = split_if_needed(X, y, test_size=0.2)
    assert len(yte) == 14 and set(yte) == set(range(7))


def test_user_image_becomes_mnist_format(synthetic_digit):
    res = preprocess_user_image(synthetic_digit)
    assert res.image28.shape == (28, 28)
    assert res.features.shape == (1, 784)
    assert 0.0 <= res.image28.min() and res.image28.max() <= 1.0
    assert res.inverted  # white background -> must be inverted to MNIST's white-on-black
    # the digit should be centred and fit within the 20x20 core like MNIST
    rows = np.where(res.image28.max(axis=1) > 0.1)[0]
    assert rows.min() >= 2 and rows.max() <= 25
    cy, cx = np.argwhere(res.image28 > 0.1).mean(axis=0)
    assert abs(cy - 13.5) < 2.5 and abs(cx - 13.5) < 2.5


def test_aspect_ratio_preserved(synthetic_digit):
    res = preprocess_user_image(synthetic_digit)
    ys, xs = np.where(res.image28 > 0.3)
    height, width = np.ptp(ys) + 1, np.ptp(xs) + 1
    assert height > 2.5 * width  # a tall thin bar stays tall and thin


def test_canvas_rgba_white_on_transparent():
    rgba = np.zeros((300, 300, 4), dtype=np.uint8)
    rgba[60:240, 140:165] = [255, 255, 255, 255]  # white stroke, transparent background
    res = preprocess_user_image(rgba, composite_on="black")
    assert not res.inverted
    assert res.image28.max() > 0.9


def test_blank_canvas_rejected():
    with pytest.raises(BlankImageError):
        preprocess_user_image(np.zeros((300, 300, 4), dtype=np.uint8), composite_on="black")
    with pytest.raises(BlankImageError):
        preprocess_user_image(np.full((100, 100), 255, dtype=np.uint8))


def test_invalid_images_rejected():
    with pytest.raises(InvalidImageError):
        preprocess_user_image(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        preprocess_user_image(np.zeros((50, 50, 7), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        preprocess_user_image(np.array([]))


def test_pil_image_input(synthetic_digit):
    res = preprocess_user_image(Image.fromarray(synthetic_digit).convert("RGB"))
    assert res.image28.shape == (28, 28)
