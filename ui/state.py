"""Cached data, the active model and per-session state.

* The dataset is loaded once per server process (``st.cache_resource``).
* The active model lives in ``st.session_state['result']``; on first visit it
  is loaded from the most recently saved model on disk (if any).
* Evaluation results are cached per model configuration, in memory and on disk,
  so switching pages never re-runs a full evaluation unnecessarily.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.config import MODELS_DIR
from src.data_loader import DatasetUnavailableError, MnistData, load_mnist, validate_dataset
from src.evaluation import evaluate_model
from src.model_training import TrainingResult, load_latest
from src.preprocessing import prepare_features


@st.cache_resource(show_spinner="Loading MNIST…")
def _load_data() -> MnistData:
    return load_mnist()


def get_data() -> MnistData | None:
    """Return the dataset, or show a helpful error and return None."""
    try:
        return _load_data()
    except DatasetUnavailableError as exc:
        st.error(f"**The MNIST dataset could not be loaded.**\n\n{exc}")
        return None


def require_data() -> MnistData:
    data = get_data()
    if data is None:
        st.stop()
    return data


@st.cache_resource(show_spinner=False)
def get_test_features() -> np.ndarray:
    return prepare_features(_load_data().x_test)


@st.cache_data(show_spinner="Validating dataset…")
def get_validation_report() -> dict:
    return validate_dataset(_load_data())


def get_result() -> TrainingResult | None:
    if "result" not in st.session_state:
        st.session_state["result"] = load_latest()
    return st.session_state["result"]


def set_result(result: TrainingResult) -> None:
    st.session_state["result"] = result


def _eval_path(key: str) -> Path:
    return MODELS_DIR / f"eval_{key}.joblib"


def get_evaluation(result: TrainingResult) -> dict:
    """Evaluate the model on the full official test set (cached)."""
    cache = st.session_state.setdefault("eval_cache", {})
    key = result.config.key()
    if key in cache:
        return cache[key]
    path = _eval_path(key)
    ev = None
    if path.exists():
        try:
            ev = joblib.load(path)
            if ev.get("trained_at") != result.trained_at:
                ev = None  # stale: model was retrained with the same settings
        except Exception:
            ev = None
    if ev is None:
        data = require_data()
        with st.spinner("Evaluating on the 10,000 test images…"):
            ev = evaluate_model(result.model, get_test_features(), data.y_test)
        ev["trained_at"] = result.trained_at
        try:
            joblib.dump(ev, path, compress=3)
        except OSError:
            pass
    cache[key] = ev
    return ev


def history() -> list:
    return st.session_state.setdefault("history", [])


def first_time(tag: str) -> bool:
    """True the first time ``tag`` is seen this session (used to play animations once)."""
    seen = st.session_state.setdefault("_seen_anim", set())
    if tag in seen:
        return False
    seen.add(tag)
    return True
