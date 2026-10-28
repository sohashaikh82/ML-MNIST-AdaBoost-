"""Visualizations: a gallery of every chart in the project, all built from real data."""

import numpy as np
import streamlit as st

from src import visualization as viz
from src.comparison import load_latest_comparison
from src.manual_adaboost import DEFAULT_LEARNERS, DEFAULT_Y, run_manual_adaboost
from ui.components import PLOTLY_CONFIG, card_title, empty_state, page_header
from ui.state import get_evaluation, get_result, require_data

page_header("Visualizations", "The dataset, the model and its results — every chart uses real data.", "viz")
data = require_data()
result = get_result()
ev = get_evaluation(result) if result else None

tabs = st.tabs(["🖼️ Data", "🌲 Model", "🎯 Results", "⚖️ Comparison", "🧮 Manual AdaBoost"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        card_title("Digit gallery", "Two random training examples of each digit")
        rng = np.random.default_rng(3)
        idx = np.concatenate([rng.choice(np.where(data.y_train == d)[0], 2, replace=False) for d in range(10)])
        st.plotly_chart(viz.digit_grid(data.x_train[idx], [str(data.y_train[i]) for i in idx], cols=5, height=330),
                        width="stretch", config=PLOTLY_CONFIG)
    with c2, st.container(border=True):
        card_title("Class distribution")
        st.plotly_chart(viz.class_distribution(np.bincount(data.y_train, minlength=10),
                                               np.bincount(data.y_test, minlength=10), height=330),
                        width="stretch", config=PLOTLY_CONFIG)
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        card_title("Pixel usage across the training set", "Fraction of images in which each pixel contains ink")
        usage = (data.x_train > 0).mean(axis=0)
        st.plotly_chart(viz.pixel_heatmap(usage, height=380, zmax=1), width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"{int((usage == 0).sum())} of 784 pixels are blank in every training image — useless as features.")
    with c2, st.container(border=True):
        card_title("Pixel variability per digit", "Standard deviation of each pixel for one digit")
        d = st.select_slider("Digit", list(range(10)), value=3, key="viz_std")
        std = (data.x_train[data.y_train == d] / 255).std(axis=0)
        st.plotly_chart(viz.pixel_heatmap(std, height=340), width="stretch", config=PLOTLY_CONFIG)

with tabs[1]:
    if result is None:
        empty_state("No model yet", "Train a model to see these charts.", link="views/model_training.py",
                    link_label="Model Training")
    else:
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            card_title("Which pixels does AdaBoost rely on?",
                       "Feature importance (impurity decrease, weighted by α) reshaped to 28 × 28")
            st.plotly_chart(viz.pixel_heatmap(result.model.feature_importances_.reshape(28, 28), height=400),
                            width="stretch", config=PLOTLY_CONFIG)
        with c2, st.container(border=True):
            card_title("Learning curve over boosting rounds")
            st.plotly_chart(viz.staged_accuracy(result.staged_test_accuracy, result.staged_train_accuracy, height=400),
                            width="stretch", config=PLOTLY_CONFIG)
        with st.container(border=True):
            card_title("Weak learner weights and errors")
            st.plotly_chart(viz.estimator_weights_fig(result.estimator_weights, result.estimator_errors),
                            width="stretch", config=PLOTLY_CONFIG)

with tabs[2]:
    if ev is None:
        empty_state("No results yet", "Train a model first.", icon="🎯")
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1, st.container(border=True):
            card_title("Confusion matrix (row %)")
            st.plotly_chart(viz.confusion_matrix_fig(ev["confusion_matrix"], normalize=True, height=480),
                            width="stretch", config=PLOTLY_CONFIG)
        with c2, st.container(border=True):
            card_title("Per-digit F1-score")
            f1 = ev["report_df"].iloc[:10]["f1-score"].to_numpy()
            st.plotly_chart(viz.class_scores_bar(f1, int(np.argmax(f1)), "F1-score", height=300), width="stretch",
                            config=PLOTLY_CONFIG)
            card_title("Performance")
            st.plotly_chart(viz.performance_fig(result.train_time_s, ev["inference_ms_per_image"], height=220),
                            width="stretch", config=PLOTLY_CONFIG)
        with st.container(border=True):
            card_title("Actual vs predicted", "First 18 misclassified test images (true → predicted)")
            wrong = np.where(ev["y_pred"] != data.y_test)[0][:18]
            st.plotly_chart(viz.digit_grid(data.x_test[wrong], [f"{data.y_test[i]} → {ev['y_pred'][i]}" for i in wrong],
                                           cols=9, title_colors=[viz.RED] * len(wrong)),
                            width="stretch", config=PLOTLY_CONFIG)

with tabs[3]:
    comp = load_latest_comparison()
    if comp is None:
        empty_state("No comparison yet", "Run the Algorithm Comparison page first.",
                    link="views/algorithm_comparison.py", link_label="Algorithm Comparison", icon="⚖️")
    else:
        cfg, df = comp
        c1, c2, c3 = st.columns(3)
        for col, m, fmt in ((c1, "Test accuracy", ".2%"), (c2, "Macro F1", ".2%"), (c3, "Training time (s)", "s")):
            with col, st.container(border=True):
                st.plotly_chart(viz.comparison_bars(df, m, height=300, fmt=fmt), width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"Latest comparison: {cfg['train_size']:,} training images, seed {cfg['random_state']}.")

with tabs[4]:
    with st.container(border=True):
        card_title("Sample weights in the manual AdaBoost example",
                   "Misclassified samples gain weight after each round (default example)")
        run = run_manual_adaboost(DEFAULT_Y, DEFAULT_LEARNERS)
        st.plotly_chart(viz.manual_weights_fig(run.rounds), width="stretch", config=PLOTLY_CONFIG)
        st.page_link("views/manual_adaboost.py", label="Edit the example interactively", icon=":material/calculate:")
