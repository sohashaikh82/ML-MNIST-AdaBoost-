"""Model Evaluation: metrics, confusion matrix and a manual metric calculation."""

import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.evaluation import cell_indices, per_class_from_cm, top_confusions
from ui.components import PLOTLY_CONFIG, callout, card_title, metric_card, page_header, require_model
from ui.state import get_evaluation, require_data

page_header("Model Evaluation", "Every number here comes from predicting the 10,000 official test images.", "eval")
data = require_data()
result = require_model()
ev = get_evaluation(result)
cm = ev["confusion_matrix"]
tag = result.trained_at

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Accuracy", f"{ev['accuracy']:.2%}", f"{ev['n_correct']:,} / {ev['n_test']:,} correct", "🎯",
                variant="dark", anim_tag=f"ev-a-{tag}")
with c2:
    metric_card("Precision (macro)", f"{ev['precision_macro']:.2%}", f"weighted: {ev['precision_weighted']:.2%}", "📌",
                anim_tag=f"ev-p-{tag}", delay=1)
with c3:
    metric_card("Recall (macro)", f"{ev['recall_macro']:.2%}", f"weighted: {ev['recall_weighted']:.2%}", "🔁",
                anim_tag=f"ev-r-{tag}", delay=2)
with c4:
    metric_card("F1-score (macro)", f"{ev['f1_macro']:.2%}", f"weighted: {ev['f1_weighted']:.2%}", "⚖️",
                variant="lime", anim_tag=f"ev-f-{tag}", delay=3)
st.caption("**Averaging:** precision, recall and F1 are first computed separately for each digit. **Macro** = simple "
           "mean of the 10 per-digit values (every digit counts equally). **Weighted** = mean weighted by the number "
           "of test images of each digit. The headline numbers use **macro** averaging.")
st.write("")

# ------------------------------------------------------------------ confusion matrix
left, right = st.columns([1.35, 1], gap="large")
with left, st.container(border=True):
    card_title("Confusion matrix", "Diagonal = correct predictions · off-diagonal = mistakes. Click a cell to inspect it.")
    normalize = st.toggle("Show row percentages (recall per digit)", value=False)
    event = st.plotly_chart(viz.confusion_matrix_fig(cm, height=520, normalize=normalize), width="stretch",
                            config=PLOTLY_CONFIG, on_select="rerun", selection_mode="points", key="cm_chart")
    try:
        pts = event.selection.points if event and event.selection else []
    except AttributeError:
        pts = []
    st.session_state.setdefault("cm_actual", 4)
    st.session_state.setdefault("cm_pred", 9)
    if pts:
        sig = (str(pts[0].get("y")), str(pts[0].get("x")))
        if sig != st.session_state.get("_cm_last_click"):  # only react to a *new* click
            st.session_state["_cm_last_click"] = sig
            try:
                st.session_state["cm_actual"], st.session_state["cm_pred"] = int(sig[0]), int(sig[1])
            except ValueError:
                pass

with right, st.container(border=True):
    card_title("Inspect a cell", "Pick any actual/predicted pair (or click the matrix)")
    a, b = st.columns(2)
    actual = a.selectbox("Actual digit", list(range(10)), key="cm_actual")
    predicted = b.selectbox("Predicted digit", list(range(10)), key="cm_pred")
    idx = cell_indices(data.y_test, ev["y_pred"], actual, predicted)
    kind = "correctly classified" if actual == predicted else "misclassified"
    st.markdown(f"**{len(idx):,}** test images of **{actual}** were predicted as **{predicted}** ({kind}).")
    if len(idx):
        show = idx[:12]
        st.plotly_chart(viz.digit_grid(data.x_test[show], [f"#{i}" for i in show], cols=4), width="stretch",
                        config=PLOTLY_CONFIG)

    card_title("Most common confusions", "Read from this model's confusion matrix")
    pairs = top_confusions(cm, 6)
    st.dataframe(pd.DataFrame([{"Actual": a_, "Predicted as": p_, "Images": n_,
                                "% of that digit": f"{n_ / cm[a_].sum():.1%}"} for a_, p_, n_ in pairs]),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ report + manual calc
left, right = st.columns([1.1, 1], gap="large")
with left, st.container(border=True):
    card_title("Classification report", "Per-digit precision, recall, F1 and support (number of test images)")
    df = ev["report_df"].copy()
    st.dataframe(df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}, na_rep="")
                 .background_gradient(subset=["f1-score"], cmap="Purples", vmin=0.8, vmax=1.0),
                 width="stretch", height=500)
    per_digit_f1 = df.iloc[:10]["f1-score"]
    st.markdown(f"Best recognised: **{per_digit_f1.idxmax()}** (F1 {per_digit_f1.max():.3f}). "
                f"Hardest: **{per_digit_f1.idxmin()}** (F1 {per_digit_f1.min():.3f}).")

with right, st.container(border=True):
    card_title("Worked calculation from the confusion matrix",
               "Manually computing one digit's metrics — and checking them against scikit-learn")
    d = st.selectbox("Digit", range(10), index=8, key="manual_digit")
    m = per_class_from_cm(cm, d)
    st.markdown(f"""
- **TP** (true positives) = cell [{d},{d}] = **{m['TP']}**
- **FP** (false positives) = column {d} total − TP = {cm[:, d].sum()} − {m['TP']} = **{m['FP']}**
- **FN** (false negatives) = row {d} total − TP = {cm[d].sum()} − {m['TP']} = **{m['FN']}**
- **TN** (true negatives) = everything else = **{m['TN']:,}**
""")
    st.latex(rf"\text{{Precision}} = \frac{{TP}}{{TP+FP}} = \frac{{{m['TP']}}}{{{m['TP']}+{m['FP']}}} = {m['precision']:.4f}")
    st.latex(rf"\text{{Recall}} = \frac{{TP}}{{TP+FN}} = \frac{{{m['TP']}}}{{{m['TP']}+{m['FN']}}} = {m['recall']:.4f}")
    st.latex(rf"F_1 = \frac{{2 \times {m['precision']:.4f} \times {m['recall']:.4f}}}{{{m['precision']:.4f} + {m['recall']:.4f}}} = {m['f1']:.4f}")
    row = ev["report_df"].iloc[d]
    ok = np.allclose([m["precision"], m["recall"], m["f1"]], [row["precision"], row["recall"], row["f1-score"]])
    (st.success if ok else st.error)(
        f"scikit-learn reports precision {row['precision']:.4f}, recall {row['recall']:.4f}, F1 {row['f1-score']:.4f} — "
        + ("the manual calculation matches." if ok else "mismatch!"))
    st.latex(rf"\text{{Accuracy}} = \frac{{\text{{trace}}(CM)}}{{\text{{total}}}} = \frac{{{int(np.trace(cm)):,}}}{{{int(cm.sum()):,}}} = {np.trace(cm) / cm.sum():.4f}")

# ------------------------------------------------------------------ explanations
with st.container(border=True):
    card_title("What the metrics mean")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Accuracy** — of all test images, how many did the model label correctly?")
        st.latex(r"\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}")
        st.markdown("**Precision** (for one digit) — when the model *says* “7”, how often is it really a 7?")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
    with c2:
        st.markdown("**Recall** (for one digit) — of all real 7s, how many did the model find?")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown("**F1-score** — the harmonic mean of precision and recall; high only if both are high.")
        st.latex(r"F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
    callout("In a 10-class problem, TP/FP/FN are defined <b>one digit at a time</b> (that digit vs all others). "
            "The 10 per-digit values are then averaged (macro or weighted).")

with st.container(border=True):
    card_title("Speed")
    s1, s2, s3 = st.columns(3)
    s1.metric("Training time", f"{result.train_time_s:.2f} s")
    s2.metric("Inference — whole test set", f"{ev['inference_time_s']:.3f} s")
    s3.metric("Inference — per image", f"{ev['inference_ms_per_image']:.4f} ms")
    st.plotly_chart(viz.performance_fig(result.train_time_s, ev["inference_ms_per_image"]), width="stretch",
                    config=PLOTLY_CONFIG)
    st.caption("Measured on this computer when the model was trained/evaluated; other hardware will differ.")
