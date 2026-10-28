"""A small, fully worked binary AdaBoost example (labels in {-1, +1}).

This is an *educational* demonstration of the formulas. It is completely
separate from the trained MNIST model, which uses scikit-learn's multiclass
SAMME variant of AdaBoost.

Formulas
--------
    w_i            = 1 / N
    eps_t          = sum_i w_i * I(h_t(x_i) != y_i)
    alpha_t        = 1/2 * ln((1 - eps_t) / eps_t)
    w_i(new)       = w_i(old) * exp(-alpha_t * y_i * h_t(x_i))
    normalise      : w_i(new) /= sum_j w_j(new)
    final H(x)     = sign( sum_t alpha_t * h_t(x) )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS_FLOOR = 1e-10  # used only to avoid division by zero when a learner is perfect

# Default example: 6 students, feature = hours studied, label = pass (+1) / fail (-1).
DEFAULT_X = [1, 2, 3, 4, 5, 6]
DEFAULT_Y = [-1, -1, +1, -1, +1, +1]
# Three simple rules ("weak learners"). Each one makes at least one mistake,
# but their weighted vote classifies all six samples correctly.
#   h1: hours > 2.5 -> +1   (wrong on x4)
#   h2: hours > 4.5 -> +1   (wrong on x3)
#   h3: hours is odd -> +1  (wrong on x1 and x6)
DEFAULT_LEARNERS = {
    "h1: hours > 2.5": [-1, -1, +1, +1, +1, +1],
    "h2: hours > 4.5": [-1, -1, -1, -1, +1, +1],
    "h3: hours is odd": [+1, -1, +1, -1, +1, -1],
}


@dataclass
class RoundResult:
    learner: str
    weights_before: np.ndarray
    predictions: np.ndarray
    misclassified: np.ndarray  # bool
    error: float
    alpha: float
    weights_unnormalized: np.ndarray
    weights_after: np.ndarray
    status: str  # "ok", "perfect", "weak" (eps >= 0.5)
    note: str = ""

    def table(self, x: list, y: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "Sample": [f"x{i + 1}" for i in range(len(y))],
            "Feature": x,
            "True y": y,
            "Prediction h(x)": self.predictions,
            "Correct?": np.where(self.misclassified, "No", "Yes"),
            "Weight before": self.weights_before,
            "exp(-α·y·h)": np.exp(-self.alpha * y * self.predictions),
            "Unnormalised": self.weights_unnormalized,
            "Weight after": self.weights_after,
        })


@dataclass
class ManualRunResult:
    rounds: list = field(default_factory=list)
    final_scores: np.ndarray | None = None
    final_predictions: np.ndarray | None = None
    final_accuracy: float = 0.0
    stopped_early: bool = False


def validate_inputs(y, predictions: dict, weights=None) -> None:
    """Raise ``ValueError`` describing exactly what is wrong with the example."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("need at least two samples")
    if not np.isin(y, [-1, 1]).all():
        raise ValueError("labels must be -1 or +1")
    if not predictions:
        raise ValueError("need at least one weak learner")
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        if pred.shape != y.shape:
            raise ValueError(f"{name}: needs one prediction per sample ({len(y)})")
        if not np.isin(pred, [-1, 1]).all():
            raise ValueError(f"{name}: predictions must be -1 or +1")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != y.shape or (w < 0).any() or not math.isclose(w.sum(), 1.0, rel_tol=1e-6):
            raise ValueError("weights must be non-negative, one per sample, and sum to 1")


def initial_weights(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("number of samples must be positive")
    return np.full(n, 1.0 / n)


def weighted_error(weights, y, predictions) -> float:
    w, y, h = (np.asarray(a, dtype=float) for a in (weights, y, predictions))
    return float(np.sum(w * (h != y)))


def learner_weight(error: float) -> float:
    """alpha = 1/2 ln((1-eps)/eps), with eps clipped away from 0 and 1."""
    e = min(max(error, EPS_FLOOR), 1 - EPS_FLOOR)
    return 0.5 * math.log((1 - e) / e)


def boosting_round(name: str, weights, y, predictions) -> RoundResult:
    """Perform one AdaBoost round and report every intermediate value."""
    w = np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.asarray(predictions, dtype=float)
    miss = h != y
    eps = weighted_error(w, y, h)
    alpha = learner_weight(eps)

    status, note = "ok", ""
    if eps == 0:
        status = "perfect"
        note = ("Error is 0: this learner classifies every sample correctly, so α would be infinite. "
                f"We clip ε to {EPS_FLOOR:g} (α ≈ {alpha:.2f}); real AdaBoost would stop here.")
    elif eps >= 0.5:
        status = "weak"
        note = (f"Error is {eps:.3f} ≥ 0.5: the learner is no better than random guessing, so α ≤ 0. "
                "AdaBoost discards such a learner (scikit-learn stops boosting).")

    unnorm = w * np.exp(-alpha * y * h)
    after = unnorm / unnorm.sum()
    return RoundResult(name, w, h.astype(int), miss, eps, alpha, unnorm, after, status, note)


def run_manual_adaboost(y, predictions: dict, x=None) -> ManualRunResult:
    """Run all rounds in order; stop when a learner is perfect or too weak."""
    validate_inputs(y, predictions)
    y = np.asarray(y)
    w = initial_weights(len(y))
    result = ManualRunResult()
    for name, pred in predictions.items():
        rnd = boosting_round(name, w, y, pred)
        if rnd.status == "weak":
            result.rounds.append(rnd)
            result.stopped_early = True
            break
        result.rounds.append(rnd)
        w = rnd.weights_after
        if rnd.status == "perfect":
            result.stopped_early = True
            break

    used = [r for r in result.rounds if r.status != "weak"]
    scores = np.zeros(len(y))
    for r in used:
        scores += r.alpha * r.predictions
    result.final_scores = scores
    result.final_predictions = np.where(scores >= 0, 1, -1)
    result.final_accuracy = float(np.mean(result.final_predictions == y))
    return result
