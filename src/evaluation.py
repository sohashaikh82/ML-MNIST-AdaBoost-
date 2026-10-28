"""Evaluation metrics computed from real test-set predictions."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .adaboost_model import decision_scores
from .config import CLASSES


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Predict the whole test set and compute every metric used in the app.

    Precision/recall/F1 are computed per class and then averaged two ways:
    * macro    -- plain mean over the 10 classes (each digit counts equally);
    * weighted -- mean weighted by how many test images each digit has.
    """
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_s = time.perf_counter() - start
    scores = decision_scores(model, X_test)

    out = {
        "y_pred": y_pred,
        "scores": scores,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "inference_time_s": float(inference_s),
        "inference_ms_per_image": float(inference_s / len(y_test) * 1000),
        "n_test": int(len(y_test)),
        "n_correct": int((y_pred == y_test).sum()),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)

    out["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=CLASSES)
    out["report_df"] = classification_report_df(y_test, y_pred)
    return out


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    total = int(len(y_true))
    if "accuracy" in df.index:  # sklearn puts accuracy in every column; keep it readable
        acc = float(rep["accuracy"])
        df.loc["accuracy"] = [np.nan, np.nan, acc, total]
    df.index = [f"Digit {i}" if i.isdigit() else i.title() for i in df.index]
    df["support"] = df["support"].astype(int)
    return df


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]:
    """Most frequent (actual, predicted, count) errors, read from the confusion matrix."""
    off = np.array(cm, copy=True)
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1]
    pairs = []
    for f in flat[:k]:
        a, p = np.unravel_index(f, off.shape)
        if off[a, p] == 0:
            break
        pairs.append((int(a), int(p), int(off[a, p])))
    return pairs


def per_class_from_cm(cm: np.ndarray, digit: int) -> dict:
    """Manually compute TP, FP, FN, TN and the metrics for one digit from the matrix.

    This is the worked calculation shown in the app (rubric: at least one
    manual metric calculation). It must agree with scikit-learn's numbers.
    """
    cm = np.asarray(cm)
    tp = int(cm[digit, digit])
    fp = int(cm[:, digit].sum() - tp)  # predicted as `digit` but actually another digit
    fn = int(cm[digit, :].sum() - tp)  # actually `digit` but predicted as another digit
    tn = int(cm.sum() - tp - fp - fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"digit": digit, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1}


def misclassified_indices(y_true, y_pred, actual: int | None = None, predicted: int | None = None) -> np.ndarray:
    mask = np.asarray(y_true) != np.asarray(y_pred)
    if actual is not None:
        mask &= np.asarray(y_true) == actual
    if predicted is not None:
        mask &= np.asarray(y_pred) == predicted
    return np.where(mask)[0]


def cell_indices(y_true, y_pred, actual: int, predicted: int) -> np.ndarray:
    """Test-set indices that fall in one confusion-matrix cell (can be the diagonal)."""
    return np.where((np.asarray(y_true) == actual) & (np.asarray(y_pred) == predicted))[0]
