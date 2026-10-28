"""Training, saving and loading AdaBoost models, plus an experiment log."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .adaboost_model import AdaBoostConfig, build_adaboost
from .config import EXPERIMENT_LOG, LATEST_MODEL_POINTER, MODELS_DIR
from .preprocessing import prepare_features, stratified_subsample

ProgressCallback = Callable[[str], None]


class TrainingError(RuntimeError):
    """Training failed; the message explains what to try next."""


@dataclass
class TrainingResult:
    """Everything measured during one real training run."""

    model: AdaBoostClassifier
    config: AdaBoostConfig
    train_time_s: float
    train_accuracy: float
    test_accuracy: float
    n_estimators_fitted: int
    baseline_test_accuracy: float  # a single tree with the same max_depth
    staged_test_accuracy: list = field(default_factory=list)
    staged_train_accuracy: list = field(default_factory=list)
    estimator_weights: list = field(default_factory=list)
    estimator_errors: list = field(default_factory=list)
    train_indices: np.ndarray | None = None
    trained_at: str = ""
    sklearn_version: str = sklearn.__version__
    source: str = "trained in this session"

    def summary(self) -> dict:
        return {
            "config_key": self.config.key(),
            **self.config.to_dict(),
            "train_time_s": round(self.train_time_s, 3),
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "baseline_test_accuracy": round(self.baseline_test_accuracy, 4),
            "n_estimators_fitted": self.n_estimators_fitted,
            "trained_at": self.trained_at,
        }


def train_adaboost(
    config: AdaBoostConfig,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    progress: ProgressCallback | None = None,
) -> TrainingResult:
    """Train AdaBoost on a stratified subset and measure it on the full test set.

    The raw uint8 images are normalised/flattened here, so the caller cannot
    accidentally pass differently preprocessed data at prediction time.
    """
    report = progress or (lambda _msg: None)
    try:
        config.validate()
    except ValueError as exc:
        raise TrainingError(f"Invalid configuration: {exc}") from exc

    report("Preparing features (normalise 0-255 -> 0-1, flatten 28x28 -> 784)")
    X_all = prepare_features(x_train_raw)
    X_test = prepare_features(x_test_raw)
    X_train, y_sub, idx = stratified_subsample(X_all, y_train, config.train_size, config.random_state)

    report(f"Fitting AdaBoost: up to {config.n_estimators} trees of depth {config.max_depth} on {len(y_sub):,} images")
    model = build_adaboost(config)
    start = time.perf_counter()
    try:
        model.fit(X_train, y_sub)
    except MemoryError as exc:
        raise TrainingError("Ran out of memory. Reduce the training sample size or max depth.") from exc
    except Exception as exc:
        raise TrainingError(f"Training failed: {exc}. Try fewer estimators or a smaller max depth.") from exc
    train_time = time.perf_counter() - start

    report("Measuring accuracy after every boosting round (staged predictions)")
    staged_test = [float(np.mean(p == y_test)) for p in model.staged_predict(X_test)]
    staged_train = [float(np.mean(p == y_sub)) for p in model.staged_predict(X_train)]

    report("Training a single decision tree with the same depth as a baseline")
    baseline = DecisionTreeClassifier(max_depth=config.max_depth, random_state=config.random_state)
    baseline.fit(X_train, y_sub)
    baseline_acc = float(baseline.score(X_test, y_test))

    n_fitted = len(model.estimators_)
    return TrainingResult(
        model=model,
        config=config,
        train_time_s=train_time,
        train_accuracy=staged_train[-1] if staged_train else float(model.score(X_train, y_sub)),
        test_accuracy=staged_test[-1] if staged_test else float(model.score(X_test, y_test)),
        n_estimators_fitted=n_fitted,
        baseline_test_accuracy=baseline_acc,
        staged_test_accuracy=staged_test,
        staged_train_accuracy=staged_train,
        estimator_weights=[float(w) for w in model.estimator_weights_[:n_fitted]],
        estimator_errors=[float(e) for e in model.estimator_errors_[:n_fitted]],
        train_indices=idx,
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def model_path_for(config: AdaBoostConfig, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"adaboost_{config.key()}.joblib"


def save_result(result: TrainingResult, models_dir: Path = MODELS_DIR, set_latest: bool = True) -> Path:
    """Save the model plus its measured metadata; optionally mark it as 'latest'."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = model_path_for(result.config, models_dir)
    payload = {
        "format": "mnist-adaboost/v1",
        "model": result.model,
        "config": result.config.to_dict(),
        "metrics": {k: getattr(result, k) for k in (
            "train_time_s", "train_accuracy", "test_accuracy", "n_estimators_fitted",
            "baseline_test_accuracy", "staged_test_accuracy", "staged_train_accuracy",
            "estimator_weights", "estimator_errors", "trained_at", "sklearn_version",
        )},
        "train_indices": result.train_indices,
    }
    joblib.dump(payload, path, compress=3)
    if set_latest:
        (models_dir / LATEST_MODEL_POINTER.name).write_text(path.name)
    return path


def load_result(path: Path) -> TrainingResult:
    """Load a model saved by :func:`save_result`.

    Only load files produced by this project: joblib/pickle files can execute
    code when loaded, so never load models from untrusted sources.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != "mnist-adaboost/v1":
        raise ValueError(f"{path.name} is not a model saved by this project")
    if not isinstance(payload.get("model"), AdaBoostClassifier):
        raise ValueError(f"{path.name} does not contain an AdaBoost model")
    m = payload["metrics"]
    saved_version = m.get("sklearn_version", "unknown")
    return TrainingResult(
        model=payload["model"],
        config=AdaBoostConfig(**payload["config"]),
        train_time_s=m["train_time_s"],
        train_accuracy=m["train_accuracy"],
        test_accuracy=m["test_accuracy"],
        n_estimators_fitted=m["n_estimators_fitted"],
        baseline_test_accuracy=m["baseline_test_accuracy"],
        staged_test_accuracy=m["staged_test_accuracy"],
        staged_train_accuracy=m["staged_train_accuracy"],
        estimator_weights=m["estimator_weights"],
        estimator_errors=m["estimator_errors"],
        train_indices=payload.get("train_indices"),
        trained_at=m["trained_at"],
        sklearn_version=saved_version,
        source=f"loaded from {path.name}",
    )


def load_latest(models_dir: Path = MODELS_DIR) -> TrainingResult | None:
    """Load the most recently saved model, or ``None`` if there is none / it is unusable."""
    pointer = Path(models_dir) / LATEST_MODEL_POINTER.name
    if not pointer.exists():
        return None
    try:
        return load_result(Path(models_dir) / pointer.read_text().strip())
    except Exception:
        return None


def load_cached_for(config: AdaBoostConfig, models_dir: Path = MODELS_DIR) -> TrainingResult | None:
    """Return a previously trained model with exactly this configuration, if saved."""
    path = model_path_for(config, models_dir)
    if not path.exists():
        return None
    try:
        result = load_result(path)
        result.source = f"reused saved model ({path.name}) -- same settings already trained"
        return result
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Experiment log (for comparing configurations)
# --------------------------------------------------------------------------- #
def read_experiments(log_path: Path = EXPERIMENT_LOG) -> list[dict]:
    try:
        return json.loads(Path(log_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def log_experiment(result: TrainingResult, log_path: Path = EXPERIMENT_LOG) -> list[dict]:
    """Append (or replace, if the same config) one run in the experiment log."""
    rows = [r for r in read_experiments(log_path) if r.get("config_key") != result.config.key()]
    rows.append(result.summary())
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(rows, indent=2))
    return rows
