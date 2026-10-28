"""AdaBoost model definition (scikit-learn) with version-compatibility handling."""

from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_SIZE,
)


@dataclass(frozen=True)
class AdaBoostConfig:
    """Hyper-parameters for one AdaBoost experiment.

    n_estimators : maximum number of boosting rounds (weak learners).
    learning_rate: shrinks each learner's vote; smaller = more cautious steps.
    max_depth    : depth of each decision-tree weak learner (1 = a "stump").
    train_size   : how many MNIST training images to use (stratified subset).
    random_state : seed so the same settings always give the same model.
    """

    n_estimators: int = DEFAULT_N_ESTIMATORS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_depth: int = DEFAULT_MAX_DEPTH
    train_size: int = DEFAULT_TRAIN_SIZE
    random_state: int = DEFAULT_RANDOM_STATE

    def validate(self) -> None:
        """Raise ``ValueError`` with a readable message if a setting is invalid."""
        if not isinstance(self.n_estimators, (int, np.integer)) or not 1 <= self.n_estimators <= 1000:
            raise ValueError("n_estimators must be an integer between 1 and 1000")
        if not 0 < float(self.learning_rate) <= 10:
            raise ValueError("learning_rate must be greater than 0 and at most 10")
        if not isinstance(self.max_depth, (int, np.integer)) or not 1 <= self.max_depth <= 50:
            raise ValueError("max_depth must be an integer between 1 and 50")
        if not isinstance(self.train_size, (int, np.integer)) or not 100 <= self.train_size <= 60000:
            raise ValueError("train_size must be an integer between 100 and 60000")
        if not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0:
            raise ValueError("random_state must be a non-negative integer")

    def key(self) -> str:
        """Short stable hash identifying this configuration (used for caching)."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]  # noqa: S324 (not security)

    def to_dict(self) -> dict:
        return asdict(self)


def build_adaboost(config: AdaBoostConfig) -> AdaBoostClassifier:
    """Create an (unfitted) multiclass AdaBoost classifier with tree weak learners.

    Handles the scikit-learn API differences:
    * ``base_estimator`` was renamed to ``estimator`` in 1.2;
    * the ``algorithm`` argument ("SAMME"/"SAMME.R") was deprecated in 1.6 and
      later removed -- SAMME is now the only algorithm. We request SAMME
      explicitly on versions that still accept the argument.
    """
    config.validate()
    weak_learner = DecisionTreeClassifier(max_depth=config.max_depth, random_state=config.random_state)
    params = inspect.signature(AdaBoostClassifier.__init__).parameters
    kwargs = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "random_state": config.random_state,
    }
    kwargs["estimator" if "estimator" in params else "base_estimator"] = weak_learner
    if "algorithm" in params:
        kwargs["algorithm"] = "SAMME"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return AdaBoostClassifier(**kwargs)


def decision_scores(model: AdaBoostClassifier, X: np.ndarray) -> np.ndarray:
    """Raw AdaBoost class scores (weighted votes) -- *not* probabilities."""
    return model.decision_function(X)


def vote_share(model: AdaBoostClassifier, X: np.ndarray) -> np.ndarray:
    """Share of the ensemble's α-weighted votes that each class receives (rows sum to 1).

        share_k(x) = Σ_t α_t · [h_t(x) = k]  /  Σ_t α_t

    This is exactly what AdaBoost's final vote adds up, so it is easy to explain
    ("54% of the weighted vote went to 7") and its arg-max is the prediction.
    It is NOT a calibrated probability: 0.6 does not mean "right 60% of the time".
    (scikit-learn's ``predict_proba`` is a softmax of the decision scores, which
    for 10 classes is very flat and harder to interpret, so we don't use it.)
    """
    n_fitted = len(model.estimators_)
    alphas = np.asarray(model.estimator_weights_[:n_fitted], dtype=float)
    votes = np.stack([est.predict(X) for est in model.estimators_])  # (T, n_samples)
    shares = np.zeros((len(X), len(model.classes_)))
    for k, cls in enumerate(model.classes_):
        shares[:, k] = ((votes == cls) * alphas[:, None]).sum(axis=0)
    total = alphas.sum()
    return shares / total if total > 0 else np.full_like(shares, 1 / len(model.classes_))
