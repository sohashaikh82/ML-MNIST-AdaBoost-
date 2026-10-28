"""Digit Predictor: draw or upload a digit and classify it with the trained AdaBoost model."""

import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

from src import visualization as viz
from src.adaboost_model import decision_scores, vote_share
from src.preprocessing import BlankImageError, InvalidImageError, preprocess_user_image
from ui.components import PLOTLY_CONFIG, callout, card_title, empty_state, page_header, require_model
from ui.state import history

try:
    import inspect

    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
    CANVAS_PARAMS = set(inspect.signature(st_canvas).parameters)
except Exception:  # component missing or incompatible
    CANVAS_AVAILABLE = False
    CANVAS_PARAMS = set()

page_header("Digit Predictor", "Draw a digit from 0 to 9 and let the AdaBoost ensemble vote.", "pred")
result = require_model()
model = result.model

st.session_state.setdefault("canvas_round", 0)
st.session_state.setdefault("last_prediction", None)


def _thumb(image28: np.ndarray, size: int = 56) -> Image.Image:
    return Image.fromarray((image28 * 255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def _predict(raw_image, source: str, composite_on: str) -> None:
    """Preprocess, classify and store the result in session state."""
    try:
        prep = preprocess_user_image(raw_image, composite_on=composite_on)
    except BlankImageError:
        st.warning("The canvas looks empty — draw a digit first (use a thick pen and fill most of the box).")
        return
    except InvalidImageError as exc:
        st.error(f"That image could not be used: {exc}")
        return

    start = time.perf_counter()
    raw = decision_scores(model, prep.features)[0]
    norm = vote_share(model, prep.features)[0]
    digit = int(model.predict(prep.features)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    entry = {"digit": digit, "raw": raw, "norm": norm, "prep": prep, "latency_ms": latency_ms,
             "source": source, "time": datetime.now().strftime("%H:%M:%S"), "thumb": _thumb(prep.image28),
             "id": f"{time.time():.6f}"}
    st.session_state["last_prediction"] = entry
    history().insert(0, entry)
    del history()[20:]
    st.toast(f"Predicted digit {digit}", icon="✅")


# ------------------------------------------------------------------ layout
left, right = st.columns([1, 1.15], gap="large")

with left, st.container(border=True):
    card_title("Input", "Draw with your mouse, touchpad or finger — or upload a photo/scan of one digit.")
    mode = st.segmented_control("Input source", ["Draw", "Upload image"], default="Draw",
                                label_visibility="collapsed", key="input_mode") or "Draw"

    if mode == "Draw":
        if not CANVAS_AVAILABLE:
            st.error("The drawing component (streamlit-drawable-canvas) is not installed. "
                     "Run `pip install -r requirements.txt`, or use **Upload image**.")
        else:
            t1, t2 = st.columns([1.2, 1])
            tool = t1.segmented_control("Tool", ["✏️ Draw", "🧽 Erase"], default="✏️ Draw", key="tool") or "✏️ Draw"
            pen = t2.slider("Pen thickness", 10, 40, 22, key="pen")
            erasing = tool.endswith("Erase")
            canvas_kwargs = dict(
                fill_color="rgba(0,0,0,0)",
                stroke_width=int(pen * (1.6 if erasing else 1)),
                stroke_color="#000000" if erasing else "#FFFFFF",
                background_color="#000000",
                height=300, width=300,
                drawing_mode="freedraw",
                update_streamlit=True,
                key=f"canvas_{st.session_state['canvas_round']}",
            )
            # The component's API changed between releases: 0.13+ needs return_image_data,
            # older releases accept display_toolbar instead. Pass only what this version supports.
            if "return_image_data" in CANVAS_PARAMS:
                canvas_kwargs["return_image_data"] = True
            if "display_toolbar" in CANVAS_PARAMS:
                canvas_kwargs["display_toolbar"] = True
            canvas = st_canvas(**canvas_kwargs)
            st.caption("Tip: draw large and centred, filling most of the box. Use the canvas toolbar to undo.")
            b1, b2 = st.columns(2)
            if b1.button("Predict Digit", type="primary", icon=":material/bolt:", width="stretch"):
                try:
                    image_data = canvas.image_data
                except Exception:
                    image_data = None
                if image_data is None:
                    st.warning("The canvas has not loaded yet — draw a digit and try again.")
                else:
                    _predict(np.asarray(image_data).astype(np.uint8), "drawing", composite_on="black")
            if b2.button("Clear canvas", icon=":material/ink_eraser:", width="stretch"):
                st.session_state["canvas_round"] += 1
                st.rerun()
    else:
        upload = st.file_uploader("Upload an image of a single digit", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                  key=f"upload_{st.session_state['canvas_round']}")
        if upload is not None:
            try:
                img = Image.open(upload)
                img.load()
                st.image(img, caption="Uploaded image", width=180)
            except Exception:
                img = None
                st.error("That file is not a readable image. Try a PNG or JPG.")
            if img is not None and st.button("Predict Digit", type="primary", icon=":material/bolt:"):
                _predict(img, "upload", composite_on="white")
        callout("Works best with one dark digit on a light background (or light on dark), "
                "cropped reasonably close. The app inverts, crops and centres it automatically.")

with right:
    pred = st.session_state["last_prediction"]
    if pred is None:
        with st.container(border=True):
            empty_state("Waiting for a digit", "Draw a digit on the left and press <b>Predict Digit</b>.", icon="✍️")
    else:
        raw, norm, digit = pred["raw"], pred["norm"], pred["digit"]
        order = np.argsort(raw)[::-1]
        runner_up = int(order[1])
        margin = float(norm[digit] - norm[runner_up])  # difference in vote share
        st.markdown(
            f"""<div class="result-card reveal" id="r{pred['id'].replace('.', '')}">
            <div class="r-label">Predicted digit</div><div class="r-digit">{digit}</div>
            <div class="r-meta">Weighted vote share <b>{norm[digit]:.1%}</b> · runner-up {runner_up}
            · latency {pred['latency_ms']:.1f} ms</div></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.container(border=True):
            tab_norm, tab_raw = st.tabs(["Weighted vote share", "Raw decision scores"])
            with tab_norm:
                st.plotly_chart(viz.class_scores_bar(norm, digit, "Share of weighted votes", height=270),
                                width="stretch", config=PLOTLY_CONFIG)
                st.caption("Each tree votes for one digit with strength α (its reliability). This is the share of the "
                           "total α each digit received — the bars sum to 100%. **Not a calibrated probability**: use "
                           "it to compare digits, not as the chance that the answer is right.")
            with tab_raw:
                st.plotly_chart(viz.class_scores_bar(raw, digit, "Decision score", height=270),
                                width="stretch", config=PLOTLY_CONFIG)
                st.caption("scikit-learn's `decision_function` for SAMME: a vote for digit k counts +1, a vote for any "
                           "other digit counts −1/9, weighted by α and averaged. Same ranking as the vote share.")

        with st.container(border=True):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                card_title("What the model saw", "Your input after preprocessing (28 × 28)")
                st.plotly_chart(viz.single_digit(pred["prep"].image28, height=220), width="stretch",
                                config=PLOTLY_CONFIG)
            with c2:
                card_title("Why this answer?")
                strength = "clear" if margin > 0.25 else "narrow" if margin > 0.08 else "very close"
                st.markdown(
                    f"The {result.n_estimators_fitted} weighted decision trees gave digit **{digit}** the highest "
                    f"combined vote: **{norm[digit]:.0%}** of the weighted votes. The next best was **{runner_up}** "
                    f"({norm[runner_up]:.0%}), a **{strength}** margin.\n\n"
                    f"Each tree looks at pixel brightness only, so unusual stroke styles, slants or off-centre "
                    f"drawings can shift the vote. If the answer is wrong, try drawing larger and more centred."
                )
            with st.expander("See every preprocessing step"):
                steps = pred["prep"].steps
                cols = st.columns(len(steps))
                for col, (name, arr) in zip(cols, steps.items()):
                    col.image(Image.fromarray(arr).resize((96, max(1, int(96 * arr.shape[0] / max(arr.shape[1], 1))))),
                              caption=name)
                st.caption("Grayscale → invert if the background is light (MNIST is white-on-black) → remove faint "
                           "noise and crop to the digit → scale longest side to 20 px keeping aspect ratio → paste "
                           "into 28×28 and centre by centre of mass → divide by 255.")

        if st.button("Try Another Digit", icon=":material/refresh:", type="primary"):
            st.session_state["last_prediction"] = None
            st.session_state["canvas_round"] += 1
            st.rerun()

# ------------------------------------------------------------------ history
st.write("")
with st.container(border=True):
    card_title("Prediction history", "Most recent first · kept for this browser session")
    hist = history()
    if not hist:
        st.caption("No predictions yet.")
    else:
        cols = st.columns(6)
        for i, item in enumerate(hist[:12]):
            with cols[i % 6]:
                st.image(item["thumb"].resize((84, 84), Image.Resampling.NEAREST))
                st.markdown(f"**{item['digit']}** · <span class='muted' style='font-size:.75rem'>{item['time']} · "
                            f"{item['source']}</span>", unsafe_allow_html=True)
        if st.button("Clear history", icon=":material/delete_sweep:"):
            hist.clear()
            st.rerun()
