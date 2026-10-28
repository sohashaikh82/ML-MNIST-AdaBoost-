"""Evaluation-metric and manual AdaBoost calculation tests."""

import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluation import cell_indices, misclassified_indices, per_class_from_cm, top_confusions
from src.manual_adaboost import (
    DEFAULT_LEARNERS,
    DEFAULT_Y,
    boosting_round,
    initial_weights,
    learner_weight,
    run_manual_adaboost,
    validate_inputs,
    weighted_error,
)


class _FixedModel:
    """Tiny stand-in model returning fixed predictions, to test evaluate_model exactly."""

    def __init__(self, preds):
        self.preds = np.asarray(preds)

    def predict(self, X):
        return self.preds

    def decision_function(self, X):
        out = np.zeros((len(self.preds), 10))
        out[np.arange(len(self.preds)), self.preds] = 1
        return out


Y_TRUE = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9])
Y_PRED = np.array([0, 1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 4])


def test_evaluate_model_matches_sklearn():
    from src.evaluation import evaluate_model

    ev = evaluate_model(_FixedModel(Y_PRED), np.zeros((len(Y_PRED), 784)), Y_TRUE)
    assert math.isclose(ev["accuracy"], 11 / 14)
    assert math.isclose(ev["precision_macro"], precision_score(Y_TRUE, Y_PRED, average="macro", zero_division=0))
    assert math.isclose(ev["recall_weighted"], recall_score(Y_TRUE, Y_PRED, average="weighted", zero_division=0))
    assert math.isclose(ev["f1_macro"], f1_score(Y_TRUE, Y_PRED, average="macro", zero_division=0))
    assert ev["confusion_matrix"].shape == (10, 10)
    assert ev["confusion_matrix"].sum() == len(Y_TRUE)
    assert np.trace(ev["confusion_matrix"]) == ev["n_correct"] == 11
    assert ev["inference_time_s"] >= 0


def test_manual_per_class_metrics_match_sklearn():
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=list(range(10)))
    for d in range(10):
        m = per_class_from_cm(cm, d)
        assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == len(Y_TRUE)
        assert math.isclose(m["precision"], precision_score(Y_TRUE, Y_PRED, labels=[d], average="macro", zero_division=0))
        assert math.isclose(m["recall"], recall_score(Y_TRUE, Y_PRED, labels=[d], average="macro", zero_division=0))
    # digit 1: TP=2 (rows 2,3), FP=1 (a 0 predicted as 1), FN=0
    m1 = per_class_from_cm(cm, 1)
    assert (m1["TP"], m1["FP"], m1["FN"]) == (2, 1, 0)
    assert math.isclose(m1["precision"], 2 / 3) and m1["recall"] == 1.0
    assert math.isclose(m1["f1"], 0.8)


def test_top_confusions_and_indices():
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=list(range(10)))
    pairs = top_confusions(cm, 10)
    assert len(pairs) == 3 and all(n == 1 for *_, n in pairs)
    assert {(a, p) for a, p, _ in pairs} == {(0, 1), (2, 3), (9, 4)}
    assert list(misclassified_indices(Y_TRUE, Y_PRED)) == [1, 5, 13]
    assert list(misclassified_indices(Y_TRUE, Y_PRED, actual=9)) == [13]
    assert list(cell_indices(Y_TRUE, Y_PRED, 1, 1)) == [2, 3]


# ----------------------------------------------------------------- manual AdaBoost
def test_initial_weights():
    w = initial_weights(6)
    assert np.allclose(w, 1 / 6) and math.isclose(w.sum(), 1.0)
    with pytest.raises(ValueError):
        initial_weights(0)


def test_round_one_hand_calculation():
    w = initial_weights(6)
    h1 = DEFAULT_LEARNERS["h1: hours > 2.5"]
    eps = weighted_error(w, DEFAULT_Y, h1)
    assert math.isclose(eps, 1 / 6)
    alpha = learner_weight(eps)
    assert math.isclose(alpha, 0.5 * math.log(5))  # 0.8047
    r = boosting_round("h1", w, DEFAULT_Y, h1)
    # after normalising: correct samples 0.1, the one mistake (x4) 0.5
    assert np.allclose(r.weights_after, [0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
    assert math.isclose(r.weights_after.sum(), 1.0)


def test_full_example_ensemble_beats_each_learner():
    run = run_manual_adaboost(DEFAULT_Y, DEFAULT_LEARNERS)
    assert len(run.rounds) == 3 and not run.stopped_early
    assert math.isclose(run.rounds[1].error, 0.1)
    assert math.isclose(run.rounds[1].alpha, 0.5 * math.log(9))
    assert run.final_accuracy == 1.0
    for r in run.rounds:
        assert np.mean(r.predictions == np.array(DEFAULT_Y)) < 1.0


def test_edge_case_perfect_learner():
    y = [1, -1, 1, -1]
    run = run_manual_adaboost(y, {"perfect": y, "later": [1, 1, 1, 1]})
    assert run.rounds[0].status == "perfect" and run.stopped_early
    assert len(run.rounds) == 1 and math.isfinite(run.rounds[0].alpha)
    assert run.final_accuracy == 1.0


def test_edge_case_weak_learner():
    y = [1, 1, -1, -1]
    run = run_manual_adaboost(y, {"coin flip": [1, -1, 1, -1]})
    assert run.rounds[0].status == "weak" and math.isclose(run.rounds[0].alpha, 0.0, abs_tol=1e-12)
    worse = boosting_round("worse", initial_weights(4), y, [-1, -1, 1, -1])
    assert worse.status == "weak" and worse.alpha < 0


@pytest.mark.parametrize("y, preds", [
    ([1, 0, -1], {"h": [1, 1, 1]}),            # label 0 not allowed
    ([1, -1], {"h": [1, -1, 1]}),              # wrong length
    ([1, -1], {"h": [1, 2]}),                  # prediction 2 not allowed
    ([1], {"h": [1]}),                         # too few samples
    ([1, -1], {}),                             # no learners
])
def test_invalid_manual_inputs(y, preds):
    with pytest.raises(ValueError):
        validate_inputs(y, preds)


def test_invalid_weights():
    with pytest.raises(ValueError):
        validate_inputs([1, -1], {"h": [1, -1]}, weights=[0.7, 0.7])
