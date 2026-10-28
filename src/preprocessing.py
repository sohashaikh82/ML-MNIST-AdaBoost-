"""Preprocessing for the dataset and for user-drawn / uploaded digits.

Dataset pipeline
----------------
raw uint8 images (0-255, 28x28) -> normalize to float32 in [0, 1]
-> flatten to 784-feature vectors -> optional stratified training subset.

Normalisation is a fixed division by 255, so nothing is "fitted" on the data
and there is no risk of test-set leakage.

User-image pipeline
-------------------
Arbitrary images are converted into the same format MNIST uses: a white digit
on a black background, cropped, scaled so its longest side is 20 px (aspect
ratio preserved), pasted into a 28x28 frame and centred by centre of mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split

from .config import IMAGE_SIZE, N_FEATURES

# --------------------------------------------------------------------------- #
# Dataset preprocessing
# --------------------------------------------------------------------------- #

def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Scale raw pixel values from 0-255 to 0.0-1.0 (float32)."""
    images = np.asarray(images)
    if images.size and (images.min() < 0 or images.max() > 255):
        raise ValueError("pixel values must be in the range 0-255 before normalisation")
    return images.astype(np.float32) / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Turn (n, 28, 28) images into (n, 784) feature vectors.

    Scikit-learn estimators expect a 2-D table: one row per sample, one column
    per feature. Each pixel therefore becomes one feature.
    """
    images = np.asarray(images)
    if images.ndim == 2 and images.shape == (IMAGE_SIZE, IMAGE_SIZE):
        images = images[None, ...]
    if images.ndim != 3 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected images of shape (n, 28, 28), got {images.shape}")
    return images.reshape(len(images), N_FEATURES)


def prepare_features(images: np.ndarray) -> np.ndarray:
    """Normalise then flatten -- the exact transform used for training and prediction."""
    return flatten_images(normalize_pixels(images))


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a reproducible, class-balanced subset of the training data.

    Returns (X_subset, y_subset, indices). If ``n_samples`` >= len(y) the full
    data is returned unchanged.
    """
    n_total = len(y)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_samples >= n_total:
        idx = np.arange(n_total)
        return X, y, idx
    idx, _ = train_test_split(
        np.arange(n_total), train_size=n_samples, stratify=y, random_state=random_state
    )
    idx = np.sort(idx)
    return X[idx], y[idx], idx


def split_if_needed(X: np.ndarray, y: np.ndarray, test_size: float = 1 / 7, random_state: int = 42):
    """Stratified train/test split for datasets that come *without* an official split.

    MNIST already has one (60k/10k) and the app always keeps it; this helper
    exists for completeness and is covered by the tests.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# --------------------------------------------------------------------------- #
# User image preprocessing
# --------------------------------------------------------------------------- #

class BlankImageError(ValueError):
    """The image contains no visible digit."""


class InvalidImageError(ValueError):
    """The input could not be interpreted as an image."""


@dataclass
class UserImageResult:
    """Output of :func:`preprocess_user_image`, including each intermediate step."""

    image28: np.ndarray  # float32 (28, 28) in [0, 1] -- what the model sees
    features: np.ndarray  # float32 (1, 784)
    steps: dict = field(default_factory=dict)  # name -> uint8 array, for display
    inverted: bool = False
    bbox: tuple | None = None


FOREGROUND_THRESHOLD = 40  # 0-255; pixels brighter than this count as "ink"
MIN_INK_PIXELS = 15  # fewer ink pixels than this -> treat as blank


def _to_grayscale_array(image, composite_on: str = "white") -> np.ndarray:
    """Convert a PIL image / numpy array (gray, RGB or RGBA) to uint8 grayscale."""
    if isinstance(image, Image.Image):
        pil = image
    else:
        arr = np.asarray(image)
        if arr.size == 0 or arr.ndim not in (2, 3):
            raise InvalidImageError(f"unsupported image array shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255 if arr.max() > 1 else 1)
            arr = (arr * (255 if arr.max() <= 1 else 1)).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            raise InvalidImageError(f"unsupported channel count {arr.shape[2]}")
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        pil = Image.fromarray(arr)

    if pil.width < 8 or pil.height < 8:
        raise InvalidImageError("image is too small (minimum 8x8 pixels)")

    if pil.mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGBA")
        background = Image.new("RGBA", pil.size, (0, 0, 0, 255) if composite_on == "black" else (255,) * 4)
        pil = Image.alpha_composite(background, pil)
    return np.asarray(pil.convert("L"), dtype=np.uint8)


def preprocess_user_image(image, composite_on: str = "white") -> UserImageResult:
    """Convert any drawn or uploaded digit image into an MNIST-style 28x28 input.

    Args:
        image: PIL image or numpy array (grayscale, RGB or RGBA).
        composite_on: background used for transparent pixels. The drawing
            canvas paints white strokes on transparency, so it passes "black";
            uploads default to "white".

    Raises:
        InvalidImageError: unreadable / unsupported input.
        BlankImageError: no digit could be found.
    """
    steps: dict = {}
    gray = _to_grayscale_array(image, composite_on=composite_on)
    steps["1. Grayscale"] = gray.copy()

    # Decide the background colour from the image border: MNIST is light ink on
    # a dark background, so a light border means we must invert.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    inverted = bool(np.median(border) > 127)
    if inverted:
        gray = 255 - gray
    steps["2. Inverted" if inverted else "2. No inversion needed"] = gray.copy()

    # Remove faint background noise, then stretch contrast so the ink is bright.
    gray = np.where(gray > FOREGROUND_THRESHOLD, gray, 0).astype(np.uint8)
    ink = gray > 0
    if ink.sum() < MIN_INK_PIXELS:
        raise BlankImageError("no digit found -- the canvas/image looks empty")
    gray = (gray.astype(np.float32) * (255.0 / gray.max())).astype(np.uint8)

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    cropped = gray[top:bottom, left:right]
    steps["3. Cropped to digit"] = cropped.copy()

    # Scale so the longest side is 20 px, preserving aspect ratio (like MNIST).
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(
        Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    steps["4. Resized (aspect kept)"] = resized.copy()

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    y0, x0 = (IMAGE_SIZE - new_h) // 2, (IMAGE_SIZE - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # Centre by centre of mass, as in the original MNIST preparation.
    cy, cx = ndimage.center_of_mass(canvas)
    if np.isfinite(cy) and np.isfinite(cx):
        canvas = ndimage.shift(canvas, (IMAGE_SIZE / 2 - cy - 0.5, IMAGE_SIZE / 2 - cx - 0.5), order=1, mode="constant")
    canvas = np.clip(canvas, 0, 255)
    steps["5. Centred 28x28"] = canvas.astype(np.uint8)

    image28 = (canvas / 255.0).astype(np.float32)
    return UserImageResult(
        image28=image28,
        features=image28.reshape(1, N_FEATURES),
        steps=steps,
        inverted=inverted,
        bbox=(int(top), int(bottom), int(left), int(right)),
    )
