"""How the Model Thinks: boosting concepts, what the trees look at, and real mistakes."""

import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.evaluation import misclassified_indices, top_confusions
from src.model_training import read_experiments
from ui.components import PLOTLY_CONFIG, callout, card_title, page_header, require_model
from ui.state import get_evaluation, require_data

page_header("How the Model Thinks", "What AdaBoost actually computes — and where it goes wrong.", "explain")
data = require_data()
result = require_model()
ev = get_evaluation(result)
model = result.model

callout("<b>Measured vs interpretation:</b> numbers, images and charts on this page are measured from the trained "
        "model. Sentences marked <i>Interpretation</i> are our reading of those numbers. The model does not "
        "'understand' digits like a person — it compares pixel brightness values against learned thresholds.", "warn")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    card_title("What is a weak learner?")
    st.markdown(f"""
A weak learner is a model only slightly better than guessing. Here each one is a **decision tree of depth
{result.config.max_depth}** — it can ask at most {result.config.max_depth} questions like *“is pixel (row 14, col 10)
brighter than 0.3?”* before choosing a digit. On its own such a tree reached **{result.baseline_test_accuracy:.1%}**
test accuracy. AdaBoost combined **{result.n_estimators_fitted}** of them to reach **{result.test_accuracy:.1%}**.
""")
    st.markdown("**Boosting vs bagging**")
    st.markdown("""
| | Boosting (AdaBoost) | Bagging (Random Forest) |
|---|---|---|
| Trees built | one after another | independently, in parallel |
| Each new tree | focuses on previous mistakes (re-weighted samples) | sees a random bootstrap sample |
| Votes | weighted by accuracy (α) | equal votes |
| Mainly reduces | bias (underfitting) | variance (overfitting) |
""")
with c2, st.container(border=True):
    card_title("Why focus on misclassified examples?")
    st.markdown("""
After each round, images the current tree got wrong get **more weight**, so the next tree is forced to find a rule
that handles them. Easy images (a clean “1”) stop dominating, and the ensemble gradually covers the awkward cases
(a “4” with a closed top, a “7” with a crossbar).

**How the trees combine:** for a new image every tree votes for one digit with strength α. The digit with the largest
total vote wins. Accurate trees (low ε) have large α and count more.

**Why handwriting is hard for this model:** digits vary in thickness, slant, size and style, and similar shapes share
pixels (4/9, 3/5, 7/9). Trees treat the 784 pixels as independent numbers — a digit shifted by two pixels looks
completely different to them, whereas a person (or a CNN) recognises the same shape.
""")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    card_title("Where do the trees look?", "Measured: feature importance of each pixel across all weighted trees")
    st.plotly_chart(viz.pixel_heatmap(model.feature_importances_.reshape(28, 28), height=380), width="stretch",
                    config=PLOTLY_CONFIG)
    st.caption("Interpretation: important pixels sit in the centre where strokes of different digits differ "
               "(e.g. the gap that separates 4 from 9). Border pixels are never used.")
with c2, st.container(border=True):
    card_title("The first question each tree asks", "Measured: root split of the first 8 weak learners")
    rows = []
    for i, tree in enumerate(model.estimators_[:8]):
        f = int(tree.tree_.feature[0])
        if f < 0:
            continue
        rows.append({"Tree": i + 1, "Pixel (row, col)": f"({f // 28}, {f % 28})",
                     "Question": f"pixel ≤ {tree.tree_.threshold[0]:.3f} ?", "α (vote weight)": round(result.estimator_weights[i], 3),
                     "Weighted error ε": round(result.estimator_errors[i], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    mask = np.zeros(784)
    for tree in model.estimators_[:8]:
        if tree.tree_.feature[0] >= 0:
            mask[tree.tree_.feature[0]] += 1
    mean_img = data.x_train[:2000].mean(axis=0) / 255
    overlay = np.clip(mean_img * 0.45 + mask.reshape(28, 28), 0, 1)
    st.plotly_chart(viz.pixel_heatmap(overlay, height=260, zmax=1), width="stretch", config=PLOTLY_CONFIG)
    st.caption("Bright dots = root-split pixels, drawn over the average digit.")

# ------------------------------------------------------------------ mistakes
with st.container(border=True):
    card_title("Inspect real mistakes", "Misclassified test images with the model's class scores")
    pairs = top_confusions(ev["confusion_matrix"], 8)
    options = ["Any mistake"] + [f"{a} predicted as {p} ({n} images)" for a, p, n in pairs]
    choice = st.selectbox("Error type (most frequent first — computed from the confusion matrix)", options)
    if choice == "Any mistake":
        idx = misclassified_indices(data.y_test, ev["y_pred"])
    else:
        a, p, _ = pairs[options.index(choice) - 1]
        idx = misclassified_indices(data.y_test, ev["y_pred"], a, p)
    st.caption(f"{len(idx):,} matching test images. Showing up to 4.")
    for i in idx[:4]:
        c1, c2, c3 = st.columns([0.8, 1.6, 1.4], vertical_alignment="center")
        scores = ev["scores"][i]
        with c1:
            st.plotly_chart(viz.single_digit(data.x_test[i], height=150), width="stretch", config=PLOTLY_CONFIG,
                            key=f"mis_img_{i}")
        with c2:
            st.plotly_chart(viz.class_scores_bar(scores, int(ev["y_pred"][i]), "Score", height=170), width="stretch",
                            config=PLOTLY_CONFIG, key=f"mis_bar_{i}")
        with c3:
            t, pr = int(data.y_test[i]), int(ev["y_pred"][i])
            st.markdown(f"**Test image #{i}**  \nActual: **{t}** · Predicted: **{pr}**  \n"
                        f"Score for {pr}: {scores[pr]:.3f} · score for {t}: {scores[t]:.3f}  \n"
                        f"True digit ranked **#{int((scores > scores[t]).sum()) + 1}** of 10.")

# ------------------------------------------------------------------ parameters
with st.container(border=True):
    card_title("How training parameters affect results")
    exps = read_experiments()
    if len(exps) >= 2:
        df = pd.DataFrame(exps)[["n_estimators", "max_depth", "learning_rate", "train_size", "test_accuracy", "train_accuracy",
                                 "train_time_s"]]
        st.dataframe(df.style.format({"test_accuracy": "{:.2%}", "train_accuracy": "{:.2%}", "train_time_s": "{:.1f}s"}),
                     hide_index=True, width="stretch")
        st.caption("Measured: your logged training runs.")
    else:
        st.caption("Train two or more configurations on the Model Training page to see measured effects here.")
    st.markdown("""
- **More estimators** → accuracy rises quickly at first and then levels off (see the staged-accuracy curve); time grows linearly.
- **Deeper weak learners** → each tree is stronger, so accuracy improves a lot from depth 1 to ~8, but deep trees overfit
  and train slower.
- **Learning rate** → small values need more trees; very large values make α jumpy and can hurt accuracy.
- **More training images** → better generalisation (smaller train–test gap), longer training.
""")
    callout("<b>Limitations:</b> pixel-based trees ignore spatial structure, so shifted, rotated, very thin or very thick "
            "digits are often misread; AdaBoost is sensitive to mislabelled images; drawings made with a mouse look "
            "different from scanned pen strokes; and the vote share is not a calibrated probability.")
