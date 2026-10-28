"""Dataset Explorer: understand MNIST and the preprocessing applied before training."""

import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.preprocessing import flatten_images, normalize_pixels, stratified_subsample
from ui.components import PLOTLY_CONFIG, callout, card_title, metric_card, page_header
from ui.state import get_validation_report, require_data

page_header("Dataset Explorer", "Look inside MNIST: real images, real counts, real pixel values.", "data")
data = require_data()
report = get_validation_report()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Training images", f"{data.n_train:,}", "Official MNIST training split", "🏋️", anim_tag="ds-train")
with c2:
    metric_card("Test images", f"{data.n_test:,}", "Official split, never used for training", "🧪", variant="dark", anim_tag="ds-test", delay=1)
with c3:
    metric_card("Image size", "28 × 28", "= 784 grayscale pixels per image", "🖼️", anim_tag="ds-size", delay=2)
with c4:
    missing = report["missing_train_pixels"] + report["missing_test_pixels"] + report["missing_labels"]
    metric_card("Missing values", f"{missing:,}", "Measured, not assumed — see Data quality", "🔍",
                variant="lime", anim_tag="ds-miss", delay=3)

st.write("")
tabs = st.tabs(["📘 Overview", "🖼️ Browse digits", "📊 Class distribution", "🔬 Pixel inspector",
                "✍️ Handwriting variation", "🧹 Data quality", "⚙️ Preprocessing"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        card_title("What is MNIST?")
        st.markdown(f"""
**MNIST** (Modified National Institute of Standards and Technology) is a collection of **{data.n_train + data.n_test:,}
handwritten digits** written by about 500 different people — US Census Bureau employees and high-school students.
It is the classic first benchmark for image classification.

- **Task:** look at an image and say which digit (0–9) it shows → a **10-class classification** problem.
- **Input features (X):** the 784 pixel brightness values of a 28 × 28 grayscale image.
  Raw values go from **0 (black background)** to **255 (white ink)**.
- **Target variable (y):** the digit label, an integer from 0 to 9.
- **Split:** {data.n_train:,} training images and {data.n_test:,} test images, written by *different* people,
  so the test score shows how well the model handles handwriting it has never seen.
- **Preparation:** each digit was size-normalised to fit a 20 × 20 box and centred by its centre of mass in the 28 × 28
  frame — our drawing pipeline copies this exactly.

Data source in this app: **{data.source}**.
""")
    with right, st.container(border=True):
        card_title("Class labels", "Each label simply is the digit drawn in the image")
        means = [data.x_train[data.y_train == d].mean() / 255 for d in range(10)]
        st.dataframe(pd.DataFrame({
            "Label": range(10),
            "Meaning": [f"handwritten '{d}'" for d in range(10)],
            "Train images": np.bincount(data.y_train, minlength=10),
            "Test images": np.bincount(data.y_test, minlength=10),
            "Avg. ink (mean pixel)": np.round(means, 3),
        }), hide_index=True, width="stretch")
        st.caption("'Avg. ink' is the mean normalised pixel value of all images of that digit — "
                   "'1' uses the least ink, '0' and '8' the most.")

# ------------------------------------------------------------------ browse
with tabs[1]:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 0.8], vertical_alignment="bottom")
        split = c1.radio("Split", ["Training", "Test"], horizontal=True, key="br_split")
        digit = c2.selectbox("Digit class", ["All digits"] + list(range(10)), key="br_digit")
        n = c3.slider("Number of samples", 6, 48, 18, step=6, key="br_n")
        if c4.button("New random set", icon=":material/shuffle:"):
            st.session_state["br_seed"] = int(np.random.default_rng().integers(1_000_000))
        X, y = (data.x_train, data.y_train) if split == "Training" else (data.x_test, data.y_test)
        pool = np.arange(len(y)) if digit == "All digits" else np.where(y == digit)[0]
        rng = np.random.default_rng(st.session_state.get("br_seed", 0))
        idx = rng.choice(pool, min(n, len(pool)), replace=False)
        st.plotly_chart(viz.digit_grid(X[idx], [f"#{i} · {y[i]}" for i in idx], cols=6), width="stretch",
                        config=PLOTLY_CONFIG)
        st.caption("Titles show the image index and its true label. Hover to read pixel values.")

# ------------------------------------------------------------------ distribution
with tabs[2]:
    left, right = st.columns([1.5, 1])
    tr, te = np.bincount(data.y_train, minlength=10), np.bincount(data.y_test, minlength=10)
    with left, st.container(border=True):
        card_title("Images per digit")
        st.plotly_chart(viz.class_distribution(tr, te, height=360), width="stretch", config=PLOTLY_CONFIG)
    with right, st.container(border=True):
        card_title("Is the dataset balanced?")
        share = tr / tr.sum()
        st.dataframe(pd.DataFrame({"Digit": range(10), "Train": tr, "Train %": np.round(share * 100, 2),
                                   "Test": te, "Test %": np.round(te / te.sum() * 100, 2)}),
                     hide_index=True, width="stretch")
        st.markdown(f"Most common: **{int(tr.argmax())}** ({share.max():.1%}); least common: **{int(tr.argmin())}** "
                    f"({share.min():.1%}). The classes are roughly balanced (max/min ratio "
                    f"{tr.max() / tr.min():.2f}), so accuracy is a meaningful metric, and macro averages are close "
                    "to weighted averages.")

# ------------------------------------------------------------------ pixel inspector
with tabs[3]:
    c1, c2 = st.columns([1, 2])
    with c1, st.container(border=True):
        card_title("Choose an image")
        split = st.radio("Split", ["Training", "Test"], horizontal=True, key="px_split")
        X, y = (data.x_train, data.y_train) if split == "Training" else (data.x_test, data.y_test)
        px_key = f"px_idx_{split}"

        def _random_index(n=len(y), k=px_key):
            st.session_state[k] = int(np.random.default_rng().integers(n))

        index = st.number_input("Image index", 0, len(y) - 1, 0, key=px_key)
        st.button("Random image", icon=":material/casino:", on_click=_random_index)
        img = X[index]
        st.plotly_chart(viz.single_digit(img, f"Label: {y[index]}", height=260), width="stretch", config=PLOTLY_CONFIG)
        st.metric("Non-zero (ink) pixels", f"{int((img > 0).sum())} / 784")
    with c2, st.container(border=True):
        card_title("Pixel-intensity heatmap", "Raw values 0–255. Hover any cell to read its value.")
        st.plotly_chart(viz.pixel_heatmap(img, height=430, zmax=255), width="stretch", config=PLOTLY_CONFIG)
    with st.expander("Show this image as a 28 × 28 matrix of raw numbers"):
        st.dataframe(pd.DataFrame(img), width="stretch", height=420)

    with st.container(border=True):
        card_title("Average image per digit", "Mean of every training image of each class (normalised 0–1)")
        mean_imgs = [data.x_train[data.y_train == d].mean(axis=0) / 255 for d in range(10)]
        st.plotly_chart(viz.digit_grid(mean_imgs, [f"mean '{d}'" for d in range(10)], cols=10, height=170),
                        width="stretch", config=PLOTLY_CONFIG)
        st.caption("Bright regions are where most writers put ink for that digit. The border pixels are almost always "
                   "0 — those features carry no information.")

# ------------------------------------------------------------------ variation
with tabs[4]:
    with st.container(border=True):
        d = st.select_slider("Digit to compare", options=list(range(10)), value=4, key="var_d")
        idx = np.where(data.y_train == d)[0]
        rng = np.random.default_rng(d)
        pick = rng.choice(idx, 12, replace=False)
        st.plotly_chart(viz.digit_grid(data.x_train[pick], [f"writer sample {i + 1}" for i in range(12)], cols=6),
                        width="stretch", config=PLOTLY_CONFIG)
        imgs = data.x_train[idx].astype(np.float32) / 255
        ink = (imgs > 0.2).sum(axis=(1, 2))
        cols = np.arange(28)
        mass = imgs.sum(axis=(1, 2)) + 1e-9
        top_cx = (imgs[:, :14, :].sum(axis=1) * cols).sum(axis=1) / (imgs[:, :14, :].sum(axis=(1, 2)) + 1e-9)
        bot_cx = (imgs[:, 14:, :].sum(axis=1) * cols).sum(axis=1) / (imgs[:, 14:, :].sum(axis=(1, 2)) + 1e-9)
        slant = top_cx - bot_cx
        m1, m2, m3 = st.columns(3)
        m1.metric("Ink pixels (median)", f"{int(np.median(ink))}", f"range {int(ink.min())}–{int(ink.max())}", delta_color="off")
        m2.metric("Slant: right-leaning images", f"{(slant > 1).mean():.0%}", "top half shifted >1px right", delta_color="off")
        m3.metric("Slant: left-leaning images", f"{(slant < -1).mean():.0%}", "top half shifted >1px left", delta_color="off")
        st.markdown(f"""
**Observations for digit {d}** (computed over all {len(idx):,} training images):
- Stroke thickness varies a lot — the number of ink pixels ranges from {int(ink.min())} to {int(ink.max())}.
- Writers slant differently: {(slant > 1).mean():.0%} lean right and {(slant < -1).mean():.0%} lean left.
- Shapes differ too (open vs closed loops, a crossed 7, a 1 with or without a base), which is why a single simple
  rule cannot recognise a digit and an ensemble of many trees helps.
""")
        st.caption(f"Total ink mass averaged {mass.mean():.1f} (sum of normalised pixel values).")

# ------------------------------------------------------------------ quality
with tabs[5]:
    with st.container(border=True):
        card_title("Data validation report", "Every value below was computed when the dataset loaded")
        rows = [
            ("Training images shape", str(report["train_shape"]), "✅"),
            ("Test images shape", str(report["test_shape"]), "✅"),
            ("Pixel data type", report["dtype"], "✅"),
            ("Pixel value range", f"{report['pixel_min']} – {report['pixel_max']}", "✅"),
            ("Missing (NaN) pixels — train", f"{report['missing_train_pixels']:,}", "✅" if not report["missing_train_pixels"] else "⚠️"),
            ("Missing (NaN) pixels — test", f"{report['missing_test_pixels']:,}", "✅" if not report["missing_test_pixels"] else "⚠️"),
            ("Missing labels", f"{report['missing_labels']:,}", "✅" if not report["missing_labels"] else "⚠️"),
            ("Labels found", ", ".join(map(str, report["labels"])), "✅" if report["labels_ok"] else "⚠️"),
            ("Exact duplicate images in training set", f"{report['duplicate_train_images']:,}", "ℹ️"),
            ("Exact duplicate images in test set", f"{report['duplicate_test_images']:,}", "ℹ️"),
            ("Identical images in both train and test", f"{report['train_test_overlap']:,}",
             "✅" if not report["train_test_overlap"] else "⚠️"),
            ("Structural problems", "none" if not report["problems"] else "; ".join(report["problems"]),
             "✅" if not report["problems"] else "⚠️"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Check", "Result", "Status"]), hide_index=True, width="stretch")
        missing = report["missing_train_pixels"] + report["missing_test_pixels"] + report["missing_labels"]
        if missing == 0:
            callout("<b>Missing-value handling:</b> the check found <b>no missing values</b>, so no imputation is "
                    "needed. (If any were found, the right fix for images would be to drop the affected samples "
                    "rather than invent pixel values.) The check stays in the pipeline so a corrupted download "
                    "would be caught.", "lime")
        else:
            callout(f"<b>{missing:,} missing values were found.</b> Affected images should be removed before training.", "warn")
        callout("<b>Duplicates:</b> exact duplicates are counted by hashing every image. A train/test overlap would "
                "inflate the test score (leakage); the count above shows whether that happens here.")

# ------------------------------------------------------------------ preprocessing
with tabs[6]:
    left, right = st.columns([1.1, 1])
    with left, st.container(border=True):
        card_title("Preprocessing pipeline (used for training and prediction)")
        st.markdown("""
1. **Load** the official split (60,000 / 10,000) — the test set is kept untouched.
2. **Validate** shapes, labels, pixel range; **check missing values and duplicates** (see Data quality).
3. **Normalise** pixels: divide by 255 so values lie in **0.0 – 1.0** instead of 0 – 255.
4. **Flatten** each 28 × 28 image into a **784-number row** (one feature per pixel).
5. **Subsample** the training set (optional, stratified so every digit keeps its share) to keep AdaBoost fast.
6. **Fix the random seed** so the same subset and the same model are produced every time.
""")
        st.code("""from src.preprocessing import prepare_features, stratified_subsample

X_train = prepare_features(x_train)        # (60000, 784) floats in [0, 1]
X_test  = prepare_features(x_test)         # (10000, 784) – same transform
X_sub, y_sub, idx = stratified_subsample(X_train, y_train,
                                         n_samples=10_000, random_state=42)""", language="python")
        callout("<b>Why flatten?</b> scikit-learn models take a 2-D table (samples × features). A decision tree "
                "asks questions such as “is pixel 350 brighter than 0.4?”, so each pixel must be its own column. "
                "Flattening loses the explicit 2-D neighbourhood, which is one reason CNNs beat tree models on images.")
        callout("<b>Why 0–1 instead of 0–255?</b> It is the same information on a smaller scale. Trees are not "
                "sensitive to scale, but normalising keeps training and drawing inputs on one consistent scale and "
                "matches what other models (and the comparison) expect. Division by a fixed 255 learns nothing from "
                "the data, so there is <b>no data leakage</b> from the test set.")
    with right, st.container(border=True):
        card_title("Before vs after preprocessing")
        i = st.number_input("Training image index", 0, data.n_train - 1, 7, key="pp_idx")
        raw = data.x_train[i]
        norm = normalize_pixels(raw)
        flat = flatten_images(norm)
        a, b = st.columns(2)
        a.markdown("**Raw (0–255)**")
        a.dataframe(pd.DataFrame(raw[10:16, 10:16]), width="stretch")
        b.markdown("**Normalised (0–1)**")
        b.dataframe(pd.DataFrame(norm[10:16, 10:16]).round(3), width="stretch")
        st.caption("A 6 × 6 crop from the centre of the image (rows/cols 10–15).")
        st.markdown(f"**Flattened shape:** `{raw.shape}` → `{flat.shape}`  \n"
                    f"First 12 non-zero features: `{np.round(flat[0][flat[0] > 0][:12], 3).tolist()}`")
        n = st.select_slider("Training subset size", [2000, 5000, 10000, 20000, 60000], value=10000, key="pp_n")
        _, ys, _ = stratified_subsample(np.zeros((data.n_train, 1)), data.y_train, n, 42)
        full = np.bincount(data.y_train, minlength=10) / data.n_train
        sub = np.bincount(ys, minlength=10) / len(ys)
        st.dataframe(pd.DataFrame({"Digit": range(10), "Full train %": np.round(full * 100, 2),
                                   f"Subset ({n:,}) %": np.round(sub * 100, 2)}).T, width="stretch")
        st.caption("Stratified sampling keeps each digit's share almost identical to the full training set.")
