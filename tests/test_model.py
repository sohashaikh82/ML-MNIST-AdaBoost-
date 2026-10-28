"""AdaBoost model, training, persistence and comparison tests (on a small real MNIST slice)."""

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.adaboost_model import AdaBoostConfig, build_adaboost, decision_scores, vote_share
from src.comparison import ComparisonConfig, best_per_metric, compare_models
from src.model_training import (
    TrainingError,
    load_cached_for,
    load_latest,
    load_result,
    log_experiment,
    read_experiments,
    save_result,
    train_adaboost,
)
from src.preprocessing import prepare_features, preprocess_user_image

SMALL = AdaBoostConfig(n_estimators=10, learning_rate=0.5, max_depth=3, train_size=2000, random_state=0)


@pytest.fixture(scope="module")
def trained(small_mnist):
    d = small_mnist
    return train_adaboost(SMALL, d.x_train, d.y_train, d.x_test, d.y_test)


def test_build_uses_decision_tree_weak_learner():
    model = build_adaboost(SMALL)
    assert isinstance(model, AdaBoostClassifier)
    weak = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
    assert isinstance(weak, DecisionTreeClassifier) and weak.max_depth == 3


@pytest.mark.parametrize("bad", [
    dict(n_estimators=0), dict(learning_rate=0), dict(learning_rate=-1), dict(max_depth=0),
    dict(train_size=10), dict(train_size=70000), dict(random_state=-1),
])
def test_invalid_parameters_rejected(bad):
    cfg = AdaBoostConfig(**{**SMALL.to_dict(), **bad})
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_config_raises_training_error(small_mnist):
    d = small_mnist
    with pytest.raises(TrainingError):
        train_adaboost(AdaBoostConfig(max_depth=0), d.x_train, d.y_train, d.x_test, d.y_test)


def test_training_produces_real_measurements(trained):
    assert trained.n_estimators_fitted >= 1
    assert 0.5 < trained.test_accuracy <= 1.0  # far above the 10% of random guessing
    assert trained.test_accuracy > trained.baseline_test_accuracy  # boosting beats a single tree
    assert len(trained.staged_test_accuracy) == trained.n_estimators_fitted
    assert np.isclose(trained.staged_test_accuracy[-1], trained.test_accuracy)
    assert trained.train_time_s > 0
    assert len(trained.estimator_weights) == trained.n_estimators_fitted


def test_training_is_reproducible(small_mnist, trained):
    d = small_mnist
    again = train_adaboost(SMALL, d.x_train, d.y_train, d.x_test, d.y_test)
    assert again.test_accuracy == trained.test_accuracy


def test_prediction_output(trained, small_mnist):
    X = prepare_features(small_mnist.x_test[:5])
    preds = trained.model.predict(X)
    assert preds.shape == (5,) and set(preds) <= set(range(10))
    raw = decision_scores(trained.model, X)
    share = vote_share(trained.model, X)
    assert raw.shape == (5, 10) and share.shape == (5, 10)
    assert np.allclose(share.sum(axis=1), 1.0) and (share >= 0).all()
    # vote share and sklearn's SAMME decision score are linked exactly: d = (K*share - 1) / (K - 1)
    assert np.allclose(raw, (10 * share - 1) / 9)
    assert np.array_equal(share.argmax(axis=1), preds)
    assert np.array_equal(raw.argmax(axis=1), preds)  # scores and predictions agree


def test_predict_user_drawing_end_to_end(trained, small_mnist):
    # Take a real test digit, scale it up and put it on a white background, like an upload.
    img = small_mnist.x_test[0]
    big = np.kron(255 - img, np.ones((8, 8), dtype=np.uint8))
    res = preprocess_user_image(big)
    pred = trained.model.predict(res.features)[0]
    assert 0 <= pred <= 9


def test_save_load_roundtrip_and_cache(trained, tmp_path):
    path = save_result(trained, tmp_path)
    loaded = load_result(path)
    assert loaded.config == trained.config
    assert loaded.test_accuracy == trained.test_accuracy
    X = np.random.rand(3, 784).astype(np.float32)
    assert np.array_equal(loaded.model.predict(X), trained.model.predict(X))
    assert load_latest(tmp_path) is not None
    assert load_cached_for(trained.config, tmp_path) is not None
    assert load_cached_for(AdaBoostConfig(n_estimators=99), tmp_path) is None


def test_missing_and_foreign_model_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_result(tmp_path / "missing.joblib")
    import joblib
    joblib.dump({"something": 1}, tmp_path / "x.joblib")
    with pytest.raises(ValueError):
        load_result(tmp_path / "x.joblib")
    assert load_latest(tmp_path / "empty") is None


def test_experiment_log(trained, tmp_path):
    log = tmp_path / "exp.json"
    log_experiment(trained, log)
    log_experiment(trained, log)  # same config replaces, not duplicates
    rows = read_experiments(log)
    assert len(rows) == 1 and rows[0]["test_accuracy"] == round(trained.test_accuracy, 4)


def test_comparison_same_split(small_mnist):
    d = small_mnist
    cfg = ComparisonConfig(train_size=1500, dt_max_depth=8, rf_n_estimators=20, ada_n_estimators=8, ada_max_depth=3)
    df = compare_models(cfg, d.x_train, d.y_train, d.x_test, d.y_test)
    assert list(df["Model"]) == ["Decision Tree", "AdaBoost", "Random Forest"]
    assert df["Test accuracy"].between(0, 1).all()
    best = best_per_metric(df)
    assert best["Test accuracy"] in set(df["Model"])
