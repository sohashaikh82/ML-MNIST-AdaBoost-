"""Loading and validating the MNIST dataset.

Loading order (first one that works wins):

1. A local cache at ``data/mnist.npz`` (shipped with the project, so the app
   works offline).
2. The Keras/TensorFlow public copy of ``mnist.npz``.
3. The original IDX files mirrored on GitHub.
4. OpenML via ``sklearn.datasets.fetch_openml`` (dataset "mnist_784").

Every download is saved to the local cache so it only happens once. The
official split (60,000 training / 10,000 test images) is always preserved.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASSES, IMAGE_SIZE, MNIST_NPZ

KERAS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
GITHUB_IDX_BASE = "https://raw.githubusercontent.com/fgnt/mnist/master/"
IDX_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


class DatasetUnavailableError(RuntimeError):
    """Raised when MNIST cannot be loaded from any source."""


@dataclass
class MnistData:
    """The MNIST images (uint8, 0-255) and labels with the official split."""

    x_train: np.ndarray  # (60000, 28, 28)
    y_train: np.ndarray  # (60000,)
    x_test: np.ndarray  # (10000, 28, 28)
    y_test: np.ndarray  # (10000,)
    source: str

    @property
    def n_train(self) -> int:
        return int(len(self.y_train))

    @property
    def n_test(self) -> int:
        return int(len(self.y_test))


# --------------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------------- #
def _load_npz(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in ("x_train", "y_train", "x_test", "y_test")}


def _download(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URLs)
        return resp.read()


def _from_keras() -> dict:
    return _load_npz(io.BytesIO(_download(KERAS_URL)))  # type: ignore[arg-type]


def _parse_idx(raw_gz: bytes, is_images: bool) -> np.ndarray:
    raw = gzip.decompress(raw_gz)
    if is_images:
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    return np.frombuffer(raw, dtype=np.uint8, offset=8)


def _from_github_idx() -> dict:
    return {
        key: _parse_idx(_download(GITHUB_IDX_BASE + name), key.startswith("x"))
        for key, name in IDX_FILES.items()
    }


def _from_openml() -> dict:
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X = X.astype(np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    y = y.astype(np.uint8)
    # OpenML stores the official training images first, then the test images.
    return {"x_train": X[:60000], "y_train": y[:60000], "x_test": X[60000:], "y_test": y[60000:]}


def load_mnist(cache_path: Path | str = MNIST_NPZ, allow_download: bool = True) -> MnistData:
    """Load MNIST, trying the local cache first and then online sources.

    Raises:
        DatasetUnavailableError: if no source works. The message explains how
        to fix it (e.g. place ``mnist.npz`` in the ``data`` folder).
    """
    cache_path = Path(cache_path)
    errors: list[str] = []

    if cache_path.exists():
        try:
            arrays = _load_npz(cache_path)
            return _to_dataclass(arrays, f"local cache ({cache_path.name})")
        except Exception as exc:  # corrupt file -> try downloading again
            errors.append(f"local cache unreadable: {exc}")

    if allow_download:
        for name, loader in (
            ("Keras public copy", _from_keras),
            ("GitHub IDX mirror", _from_github_idx),
            ("OpenML mnist_784", _from_openml),
        ):
            try:
                arrays = loader()
                data = _to_dataclass(arrays, name)
                _save_cache(arrays, cache_path)
                return data
            except Exception as exc:
                errors.append(f"{name}: {exc}")

    raise DatasetUnavailableError(
        "Could not load MNIST. Tried: " + "; ".join(errors) + ". "
        "Fix: connect to the internet and restart, or copy mnist.npz into the "
        f"'{cache_path.parent.name}' folder."
    )


def _to_dataclass(arrays: dict, source: str) -> MnistData:
    data = MnistData(
        x_train=np.asarray(arrays["x_train"]),
        y_train=np.asarray(arrays["y_train"]).astype(np.int64),
        x_test=np.asarray(arrays["x_test"]),
        y_test=np.asarray(arrays["y_test"]).astype(np.int64),
        source=source,
    )
    problems = basic_shape_problems(data)
    if problems:
        raise ValueError("dataset failed validation: " + "; ".join(problems))
    return data


def _save_cache(arrays: dict, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **arrays)
    except OSError:
        pass  # caching is a convenience; failure is not fatal


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def basic_shape_problems(data: MnistData) -> list[str]:
    """Return a list of structural problems (empty list = OK)."""
    problems = []
    for name in ("x_train", "x_test"):
        arr = getattr(data, name)
        if arr.ndim != 3 or arr.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            problems.append(f"{name} has shape {arr.shape}, expected (n, 28, 28)")
    if len(data.x_train) != len(data.y_train):
        problems.append("x_train and y_train lengths differ")
    if len(data.x_test) != len(data.y_test):
        problems.append("x_test and y_test lengths differ")
    return problems


def validate_dataset(data: MnistData) -> dict:
    """Inspect the dataset and report what was actually found.

    Checks shapes, dtypes, pixel range, missing values (NaN), label set and
    exact duplicate images. Nothing here is assumed -- every number is computed.
    """
    report: dict = {"source": data.source, "problems": basic_shape_problems(data)}
    report["train_shape"] = tuple(data.x_train.shape)
    report["test_shape"] = tuple(data.x_test.shape)
    report["dtype"] = str(data.x_train.dtype)
    report["pixel_min"] = int(min(data.x_train.min(), data.x_test.min()))
    report["pixel_max"] = int(max(data.x_train.max(), data.x_test.max()))

    # Missing values: integer arrays cannot hold NaN, but we check the float
    # view anyway so the result is measured rather than assumed.
    report["missing_train_pixels"] = int(np.isnan(data.x_train.astype(np.float32)).sum())
    report["missing_test_pixels"] = int(np.isnan(data.x_test.astype(np.float32)).sum())
    report["missing_labels"] = int(
        np.isnan(data.y_train.astype(np.float64)).sum() + np.isnan(data.y_test.astype(np.float64)).sum()
    )

    labels = sorted(set(np.unique(data.y_train)) | set(np.unique(data.y_test)))
    report["labels"] = [int(v) for v in labels]
    report["labels_ok"] = report["labels"] == CLASSES
    report["train_class_counts"] = np.bincount(data.y_train, minlength=10).tolist()
    report["test_class_counts"] = np.bincount(data.y_test, minlength=10).tolist()

    report.update(find_duplicates(data))
    return report


def _image_hashes(images: np.ndarray) -> list[str]:
    return [hashlib.md5(img.tobytes()).hexdigest() for img in images]  # noqa: S324 (not security)


def find_duplicates(data: MnistData) -> dict:
    """Count exact duplicate images inside the training set and train/test overlap."""
    train_hashes = _image_hashes(data.x_train)
    test_hashes = _image_hashes(data.x_test)
    train_set = set(train_hashes)
    return {
        "duplicate_train_images": len(train_hashes) - len(train_set),
        "duplicate_test_images": len(test_hashes) - len(set(test_hashes)),
        "train_test_overlap": sum(1 for h in set(test_hashes) if h in train_set),
    }
