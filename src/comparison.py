"""Fair comparison of Decision Tree, AdaBoost and Random Forest on the same data."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

from .adaboost_model import AdaBoostConfig, build_adaboost
from .config import MODELS_DIR
from .preprocessing import prepare_features, stratified_subsample


@dataclass(frozen=True)
class ComparisonConfig:
    train_size: int = 10_000
    random_state: int = 42
    dt_max_depth: int = 12
    rf_n_estimators: int = 100
    rf_max_depth: int = 0  # 0 = unlimited
    ada_n_estimators: int = 50
    ada_learning_rate: float = 0.5
    ada_max_depth: int = 6

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]  # noqa: S324

    def ada_config(self) -> AdaBoostConfig:
        return AdaBoostConfig(self.ada_n_estimators, self.ada_learning_rate, self.ada_max_depth,
                              self.train_size, self.random_state)


def _measure(name: str, model, X_tr, y_tr, X_te, y_te, fitted_time: float | None = None) -> dict:
    if fitted_time is None:
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fitted_time = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(X_te)
    infer = time.perf_counter() - start
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    return {
        "Model": name,
        "Test accuracy": float(accuracy_score(y_te, y_pred)),
        "Macro precision": float(p),
        "Macro recall": float(r),
        "Macro F1": float(f),
        "Training time (s)": float(fitted_time),
        "Inference time (s)": float(infer),
    }


def compare_models(
    cfg: ComparisonConfig,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Train all three models on the identical training subset and score them
    on the identical, full official test set."""
    report = progress or (lambda _m: None)
    X_all = prepare_features(x_train_raw)
    X_te = prepare_features(x_test_raw)
    X_tr, y_tr, _ = stratified_subsample(X_all, y_train, cfg.train_size, cfg.random_state)

    rows = []
    report(f"Decision Tree (max_depth={cfg.dt_max_depth})")
    rows.append(_measure("Decision Tree",
                         DecisionTreeClassifier(max_depth=cfg.dt_max_depth, random_state=cfg.random_state),
                         X_tr, y_tr, X_te, y_test))
    report(f"AdaBoost ({cfg.ada_n_estimators} trees, depth {cfg.ada_max_depth}, lr {cfg.ada_learning_rate})")
    rows.append(_measure("AdaBoost", build_adaboost(cfg.ada_config()), X_tr, y_tr, X_te, y_test))
    report(f"Random Forest ({cfg.rf_n_estimators} trees)")
    rf = RandomForestClassifier(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth or None,
                                random_state=cfg.random_state, n_jobs=-1)
    rows.append(_measure("Random Forest", rf, X_tr, y_tr, X_te, y_test))
    return pd.DataFrame(rows)


def comparison_path(cfg: ComparisonConfig, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"comparison_{cfg.key()}.json"


def save_comparison(cfg: ComparisonConfig, df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> Path:
    path = comparison_path(cfg, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "results": df.to_dict(orient="records")}, indent=2))
    (Path(models_dir) / "comparison_latest.txt").write_text(path.name)
    return path


def load_comparison(cfg: ComparisonConfig, models_dir: Path = MODELS_DIR) -> pd.DataFrame | None:
    path = comparison_path(cfg, models_dir)
    if not path.exists():
        return None
    try:
        return pd.DataFrame(json.loads(path.read_text())["results"])
    except Exception:
        return None


def load_latest_comparison(models_dir: Path = MODELS_DIR) -> tuple[dict, pd.DataFrame] | None:
    pointer = Path(models_dir) / "comparison_latest.txt"
    try:
        payload = json.loads((Path(models_dir) / pointer.read_text().strip()).read_text())
        return payload["config"], pd.DataFrame(payload["results"])
    except Exception:
        return None


def best_per_metric(df: pd.DataFrame) -> dict:
    """Which model wins each metric (higher is better, except times)."""
    best = {}
    for col in ("Test accuracy", "Macro precision", "Macro recall", "Macro F1"):
        best[col] = df.loc[df[col].idxmax(), "Model"]
    for col in ("Training time (s)", "Inference time (s)"):
        best[col] = df.loc[df[col].idxmin(), "Model"]
    return best
