"""Model Training: configure hyper-parameters and train a real AdaBoost model."""

import threading
import time

import pandas as pd
import streamlit as st

from src import visualization as viz
from src.adaboost_model import AdaBoostConfig
from src.config import SLOW_TRAINING_WARNING_SIZE
from src.model_training import (
    TrainingError,
    load_cached_for,
    log_experiment,
    read_experiments,
    save_result,
    train_adaboost,
)
from ui.components import PLOTLY_CONFIG, callout, card_title, empty_state, metric_card, page_header
from ui.state import get_result, require_data, set_result

page_header("Model Training", "Choose AdaBoost settings, train on real MNIST images and watch it learn.", "train")
data = require_data()
current = get_result()

PRESETS = {
    "⚡ Quick demo (~10 s)": AdaBoostConfig(n_estimators=30, learning_rate=0.5, max_depth=4, train_size=5000),
    "⭐ Recommended (~40 s)": AdaBoostConfig(),
    "🎯 Higher accuracy (~2–3 min)": AdaBoostConfig(n_estimators=100, learning_rate=0.5, max_depth=10, train_size=10000),
}


def _apply_preset():
    cfg = PRESETS[st.session_state["preset"]]
    for field in ("n_estimators", "learning_rate", "max_depth", "train_size", "random_state"):
        st.session_state[f"cfg_{field}"] = getattr(cfg, field)


start_cfg = current.config if current else AdaBoostConfig()
for field in ("n_estimators", "learning_rate", "max_depth", "train_size", "random_state"):
    st.session_state.setdefault(f"cfg_{field}", getattr(start_cfg, field))

left, right = st.columns([1, 1.25], gap="large")

# ------------------------------------------------------------------ configuration
with left, st.container(border=True):
    card_title("Configuration", "Times are rough guides measured on a 2-core laptop-class machine.")
    st.selectbox("Preset", list(PRESETS), index=None, placeholder="Start from a preset…", key="preset",
                 on_change=_apply_preset)
    n_est = st.slider("Number of boosting estimators (weak learners)", 1, 300, key="cfg_n_estimators",
                      help="How many decision trees AdaBoost adds one after another. More trees usually help, "
                           "up to a point, but training time grows linearly.")
    lr = st.select_slider("Learning rate", options=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
                          key="cfg_learning_rate",
                          help="Scales every tree's vote. Small values take smaller, safer steps and need more trees; "
                               "large values learn fast but can overshoot.")
    depth = st.slider("Weak learner max depth", 1, 15, key="cfg_max_depth",
                      help="How many yes/no pixel questions each tree may ask in a row. Depth 1 is a 'stump' "
                           "(one question) — too weak for 10 digits. Depth 4–8 works well for MNIST.")
    size = st.select_slider("Training sample size", options=[1000, 2000, 5000, 10000, 15000, 20000, 30000, 45000, 60000],
                            key="cfg_train_size",
                            help="A stratified random subset of the 60,000 training images. "
                                 "The test set is always the full 10,000.")
    seed = st.number_input("Random seed", 0, 99999, key="cfg_random_state",
                           help="Fixes which images are sampled and how trees break ties, so results are reproducible.")
    config = AdaBoostConfig(int(n_est), float(lr), int(depth), int(size), int(seed))

    if size >= SLOW_TRAINING_WARNING_SIZE or n_est * depth * size > 50 * 8 * 20000:
        st.warning(f"These settings may take several minutes ({size:,} images × {n_est} trees × depth {depth}). "
                   "AdaBoost trains trees one after another, so it cannot use multiple CPU cores.")
    cached = load_cached_for(config)
    force = False
    if cached is not None:
        st.info(f"A model with exactly these settings was already trained ({cached.trained_at}). "
                "It will be reused instead of training again.")
        force = st.checkbox("Retrain anyway", value=False)
    go = st.button("Train Model", type="primary", icon=":material/play_arrow:", width="stretch")

    with st.expander("What does each setting do?"):
        st.markdown("""
- **Estimators** — the number of boosting rounds. Each round adds one tree that focuses on the images earlier
  trees got wrong. scikit-learn may stop early if a tree is perfect or no better than chance.
- **Learning rate** — multiplies each tree's weight α. Lower = smoother learning, often needs more estimators.
- **Max depth** — the strength of each weak learner. Deeper trees capture more pixel combinations but are slower
  and can overfit.
- **Training sample size** — more images = better generalisation but longer training.
- **Random seed** — the same seed + settings always gives the same model (reproducibility).
""")

# ------------------------------------------------------------------ training
with right:
    if go:
        if cached is not None and not force:
            set_result(cached)
            log_experiment(cached)
            st.toast("Reused the saved model with identical settings", icon="♻️")
            st.rerun()
        holder: dict = {"messages": []}

        def _work():
            try:
                holder["result"] = train_adaboost(config, data.x_train, data.y_train, data.x_test, data.y_test,
                                                  progress=holder["messages"].append)
            except TrainingError as exc:
                holder["error"] = str(exc)
            except Exception as exc:  # unexpected — still never pretend success
                holder["error"] = f"Unexpected error: {exc}"

        worker = threading.Thread(target=_work, daemon=True)
        t0 = time.perf_counter()
        worker.start()
        with st.status("Training AdaBoost…", expanded=True) as status:
            shown = 0
            clock = st.empty()
            st.markdown('<div class="skeleton"></div><div class="skeleton" style="width:70%"></div>',
                        unsafe_allow_html=True)
            while worker.is_alive():
                while shown < len(holder["messages"]):
                    st.write(f"• {holder['messages'][shown]}")
                    shown += 1
                clock.markdown(f"⏱️ **{time.perf_counter() - t0:.1f} s elapsed** — scikit-learn does not report "
                               "per-tree progress while fitting, so this shows real elapsed time only.")
                time.sleep(0.3)
            worker.join()
            while shown < len(holder["messages"]):
                st.write(f"• {holder['messages'][shown]}")
                shown += 1
            if "error" in holder:
                status.update(label="Training failed", state="error", expanded=True)
                st.error(holder["error"])
                st.markdown("**What to try:** use a preset, lower the training sample size or max depth, "
                            "then press Train Model again. The previous model (if any) is still active.")
            else:
                res = holder["result"]
                save_result(res)
                log_experiment(res)
                set_result(res)
                status.update(label=f"Training complete in {res.train_time_s:.1f} s — "
                                    f"test accuracy {res.test_accuracy:.2%}", state="complete", expanded=False)
                st.session_state["just_trained"] = True
                st.rerun()

    result = get_result()
    if result is None:
        with st.container(border=True):
            empty_state("No model trained yet", "Pick a preset or adjust the settings, then press <b>Train Model</b>.",
                        icon="🌱")
    else:
        if st.session_state.pop("just_trained", False):
            st.toast(f"Model trained · {result.test_accuracy:.2%} test accuracy", icon="🎉")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Test accuracy", f"{result.test_accuracy:.2%}", "10,000 unseen images", "🎯", variant="dark",
                        anim_tag=f"tr-acc-{result.trained_at}")
        with c2:
            metric_card("Training accuracy", f"{result.train_accuracy:.2%}", f"on {result.config.train_size:,} training images",
                        "🏋️", anim_tag=f"tr-tr-{result.trained_at}", delay=1)
        with c3:
            metric_card("Training time", f"{result.train_time_s:.1f}s",
                        f"{result.n_estimators_fitted} of {result.config.n_estimators} learners fitted", "⏱️",
                        variant="lime", anim_tag=f"tr-time-{result.trained_at}", delay=2)
        st.write("")
        with st.container(border=True):
            card_title("Accuracy after each boosting round",
                       "Real staged predictions: the ensemble's accuracy using only the first k trees")
            st.plotly_chart(viz.staged_accuracy(result.staged_test_accuracy, result.staged_train_accuracy),
                            width="stretch", config=PLOTLY_CONFIG)
            gap = result.train_accuracy - result.test_accuracy
            st.caption(f"Train–test gap: {gap:.1%}. "
                       + ("A large gap means the model memorises the training subset (overfitting); "
                          "try smaller max depth or more training images." if gap > 0.05 else
                          "A small gap means the model generalises well to new handwriting."))
        with st.container(border=True):
            card_title("Weight (α) and error (ε) of every weak learner",
                       "Values stored by scikit-learn: estimator_weights_ and estimator_errors_")
            st.plotly_chart(viz.estimator_weights_fig(result.estimator_weights, result.estimator_errors),
                            width="stretch", config=PLOTLY_CONFIG)
        with st.container(border=True):
            card_title("Model configuration")
            cfg = result.config
            st.dataframe(pd.DataFrame({
                "Setting": ["Algorithm", "Weak learner", "Estimators requested / fitted", "Learning rate",
                            "Training images", "Test images", "Random seed", "Single-tree baseline accuracy",
                            "Status", "Trained at", "scikit-learn version"],
                "Value": ["AdaBoostClassifier (multiclass SAMME)", f"DecisionTreeClassifier(max_depth={cfg.max_depth})",
                          f"{cfg.n_estimators} / {result.n_estimators_fitted}", str(cfg.learning_rate),
                          f"{cfg.train_size:,} (stratified)", f"{data.n_test:,} (official test split)", str(cfg.random_state),
                          f"{result.baseline_test_accuracy:.2%}", result.source, result.trained_at, result.sklearn_version],
            }), hide_index=True, width="stretch")

# ------------------------------------------------------------------ experiments
st.write("")
rows = read_experiments()
with st.container(border=True):
    card_title("Compare your configurations", "Every training run is logged with its settings and measured results")
    if not rows:
        st.caption("No experiments logged yet.")
    else:
        st.plotly_chart(viz.experiments_fig(rows), width="stretch", config=PLOTLY_CONFIG)
        df = pd.DataFrame(rows)[["n_estimators", "learning_rate", "max_depth", "train_size", "random_state",
                                 "test_accuracy", "train_accuracy", "baseline_test_accuracy", "train_time_s",
                                 "n_estimators_fitted", "trained_at"]]
        df.index = [f"#{i + 1}" for i in range(len(df))]
        st.dataframe(df.style.format({"test_accuracy": "{:.2%}", "train_accuracy": "{:.2%}",
                                      "baseline_test_accuracy": "{:.2%}", "train_time_s": "{:.1f}s"})
                     .highlight_max(subset=["test_accuracy"], color="#DDF3A1"), width="stretch")
        st.caption("Tip: change one setting at a time (e.g. max depth 2 → 4 → 6 → 8) to see its real effect.")

# ------------------------------------------------------------------ algorithm explanation
with st.container(border=True):
    card_title("How AdaBoost trains — step by step")
    st.markdown("""
1. **Initialise sample weights** — every training image starts with the same weight, *wᵢ = 1/N*.
2. **Train a weak learner** — fit a small decision tree using the current weights (heavier images matter more).
3. **Measure the weighted error** *ε* — the total weight of the images this tree got wrong.
4. **Compute the learner's say** *α* — accurate trees get a larger vote. For *K* classes scikit-learn's SAMME uses
""")
    st.latex(r"\alpha_t = \eta \left[\ln\frac{1-\varepsilon_t}{\varepsilon_t} + \ln(K-1)\right]")
    st.markdown("""
   where *η* is the learning rate and *K = 10*. The extra *ln(K−1)* term means a tree only needs to beat random
   guessing among 10 classes (ε < 0.9), not 50%.
5. **Re-weight the images** — multiply the weight of each misclassified image by *e^α* and re-normalise, so the next
   tree concentrates on the hard examples.
6. **Repeat** for the chosen number of rounds (or stop early if a tree is perfect or useless).
7. **Combine** — the final prediction is the digit with the largest sum of *α* votes from all trees.
""")
    callout("The α and ε values in the chart above are the real values stored by scikit-learn for your model. "
            "Per-image weights are not stored by the library, so they are demonstrated on a small example on the "
            "<b>Manual AdaBoost</b> page rather than invented here.")
    st.page_link("views/manual_adaboost.py", label="Open the manual calculation", icon=":material/calculate:")
