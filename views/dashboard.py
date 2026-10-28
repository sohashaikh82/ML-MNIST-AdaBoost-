"""Dashboard: live overview of the dataset, the active model and recent activity."""

import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.comparison import load_latest_comparison
from ui.components import PLOTLY_CONFIG, card_title, empty_state, metric_card, page_header
from ui.state import get_evaluation, get_result, history, require_data

page_header("MNIST Intelligence Dashboard",
            "Explore, train, and evaluate AdaBoost for handwritten digit recognition.", "dash")

data = require_data()
result = get_result()
ev = get_evaluation(result) if result else None

# ---------------------------------------------------------------- metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Dataset Size", f"{data.n_train + data.n_test:,}", icon="🗂️",
                sub=f"{data.n_train:,} training · {data.n_test:,} testing<br/>Status: loaded from {data.source}",
                anim_tag="dash-size")
with c2:
    if result:
        gain = result.test_accuracy - result.baseline_test_accuracy
        metric_card("Model Accuracy", f"{result.test_accuracy:.2%}", icon="🎯", variant="dark",
                    badge=f"{gain:+.1%} vs single tree", badge_kind="up" if gain >= 0 else "warn",
                    sub=f"Test set (10,000 images). Single depth-{result.config.max_depth} tree: "
                        f"{result.baseline_test_accuracy:.1%}", anim_tag="dash-acc", delay=1)
    else:
        metric_card("Model Accuracy", "—", icon="🎯", variant="dark", sub="Train a model to measure accuracy",
                    badge="not trained", badge_kind="warn")
with c3:
    if result:
        metric_card("Training Time", f"{result.train_time_s:.1f}s", icon="⏱️",
                    badge="complete", badge_kind="neutral",
                    sub=f"{result.n_estimators_fitted} learners on {result.config.train_size:,} images · {result.source}",
                    anim_tag="dash-time", delay=2)
    else:
        metric_card("Training Time", "—", icon="⏱️", sub="No training run yet", badge="idle", badge_kind="warn")
with c4:
    st.markdown(
        '<div class="metric-card lime anim d3"><div class="m-top"><span class="m-label" style="color:#3F5600">'
        'Digit Classes</span><span class="m-icon">🔢</span></div><div class="m-value">10</div>'
        '<div class="m-sub" style="color:#3F5600">Digits 0 to 9 (multiclass)</div><div class="digits-row">'
        + "".join(f"<span>{d}</span>" for d in range(10)) + "</div></div>",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------- row 2
left, right = st.columns([1.15, 1])
with left, st.container(border=True):
    card_title("Digit-class distribution", "Real counts per digit in the official train and test splits")
    st.plotly_chart(viz.class_distribution(np.bincount(data.y_train, minlength=10),
                                           np.bincount(data.y_test, minlength=10), height=290),
                    width="stretch", config=PLOTLY_CONFIG)
with right, st.container(border=True):
    top = st.columns([2.4, 1])
    with top[0]:
        card_title("Sample handwritten digits", "Random training images with their labels")
    with top[1]:
        if st.button("Shuffle", key="dash_shuffle", icon=":material/shuffle:"):
            st.session_state["dash_seed"] = int(np.random.default_rng().integers(1_000_000))
    rng = np.random.default_rng(st.session_state.get("dash_seed", 7))
    idx = rng.choice(data.n_train, 12, replace=False)
    st.plotly_chart(viz.digit_grid(data.x_train[idx], [f"label {data.y_train[i]}" for i in idx], cols=6, height=270),
                    width="stretch", config=PLOTLY_CONFIG)

# ---------------------------------------------------------------- row 3
left, right = st.columns([1.15, 1])
with left, st.container(border=True):
    card_title("Actual vs predicted digits", "Random test images, titled actual → predicted (green = correct, red = wrong)")
    if ev is None:
        empty_state("No predictions yet", "Train a model to compare its predictions with the true labels.",
                    link="views/model_training.py", link_label="Train a model")
    else:
        rng = np.random.default_rng(st.session_state.get("dash_seed", 7) + 1)
        wrong = np.where(ev["y_pred"] != data.y_test)[0]
        right_ = np.where(ev["y_pred"] == data.y_test)[0]
        pick = np.concatenate([rng.choice(right_, 9, replace=False), rng.choice(wrong, min(3, len(wrong)), replace=False)])
        rng.shuffle(pick)
        titles = [f"{data.y_test[i]} → {ev['y_pred'][i]}" for i in pick]
        colors = [viz.LIME_DARK if data.y_test[i] == ev["y_pred"][i] else viz.RED for i in pick]
        st.plotly_chart(viz.digit_grid(data.x_test[pick], titles, cols=6, height=270, title_colors=colors),
                        width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"Overall: {ev['n_correct']:,} of {ev['n_test']:,} test images correct. "
                   "This sample deliberately includes some mistakes.")
with right, st.container(border=True):
    card_title("Confusion matrix preview", "Rows = actual digit, columns = predicted digit")
    if ev is None:
        empty_state("Nothing to show yet", "The confusion matrix appears after training.", icon="🧩")
    else:
        st.plotly_chart(viz.confusion_matrix_fig(ev["confusion_matrix"], height=300), width="stretch",
                        config=PLOTLY_CONFIG)
        st.page_link("views/model_evaluation.py", label="Open full evaluation", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- row 4
c1, c2, c3 = st.columns([1, 1, 0.9])
with c1, st.container(border=True):
    card_title("Model accuracy comparison", "Test accuracy measured on the same 10,000 images")
    comp = load_latest_comparison()
    if comp is not None:
        cfg, df = comp
        st.plotly_chart(viz.comparison_bars(df, "Test accuracy", height=260), width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"From the latest Algorithm Comparison run ({cfg['train_size']:,} training images).")
    elif result is not None:
        df = pd.DataFrame({"Model": [f"Single tree (depth {result.config.max_depth})", "AdaBoost"],
                           "Test accuracy": [result.baseline_test_accuracy, result.test_accuracy]})
        st.plotly_chart(viz.comparison_bars(df, "Test accuracy", height=260), width="stretch", config=PLOTLY_CONFIG)
        st.caption("Run Algorithm Comparison to add Decision Tree and Random Forest.")
    else:
        empty_state("No results yet", "Train a model or run a comparison.", icon="📊")
with c2, st.container(border=True):
    card_title("Training & inference performance", "Accuracy after each boosting round")
    if result is not None and result.staged_test_accuracy:
        st.plotly_chart(viz.staged_accuracy(result.staged_test_accuracy, result.staged_train_accuracy, height=230),
                        width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"Training {result.train_time_s:.1f}s · inference {ev['inference_ms_per_image']:.3f} ms per image "
                   f"({ev['inference_time_s']:.2f}s for all 10,000).")
    else:
        empty_state("No training run yet", "Performance appears after training.", icon="⏱️")
with c3, st.container(border=True):
    card_title("Recent predictions", "Your latest drawings and uploads (this session)")
    hist = history()
    if not hist:
        empty_state("No predictions yet", "Draw a digit to see it here.", icon="✏️",
                    link="views/digit_predictor.py", link_label="Open Digit Predictor")
    else:
        for item in hist[:5]:
            cols = st.columns([0.9, 3], vertical_alignment="center")
            cols[0].image(item["thumb"], width=44)
            cols[1].markdown(f"**Predicted {item['digit']}** · {item['source']}<br/>"
                             f"<span class='muted' style='font-size:.78rem'>{item['time']} · {item['latency_ms']:.1f} ms</span>",
                             unsafe_allow_html=True)
