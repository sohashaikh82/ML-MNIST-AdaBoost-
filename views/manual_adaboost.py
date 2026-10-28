"""AdaBoost: Manual Calculation Explained (binary, labels in {-1, +1})."""

import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.manual_adaboost import DEFAULT_LEARNERS, DEFAULT_X, DEFAULT_Y, run_manual_adaboost
from ui.components import PLOTLY_CONFIG, callout, card_title, page_header

page_header("AdaBoost: Manual Calculation Explained",
            "Every formula, worked by hand on six samples — then edit the example and recalculate.", "manual")

callout("<b>Scope:</b> this is a simplified <b>binary</b> AdaBoost demonstration with labels −1/+1. It is separate from "
        "the MNIST model, which uses scikit-learn's <b>multiclass</b> SAMME version (10 classes, α includes an extra "
        "ln(K−1) term). Nothing on this page comes from, or changes, the trained MNIST model.", "warn")

LEARNER_NAMES = list(DEFAULT_LEARNERS)


def _default_table() -> pd.DataFrame:
    df = pd.DataFrame({"Sample": [f"x{i + 1}" for i in range(len(DEFAULT_X))], "Hours studied": DEFAULT_X,
                       "True label y": DEFAULT_Y})
    for name, preds in DEFAULT_LEARNERS.items():
        df[name] = preds
    return df


if "manual_table" not in st.session_state:
    st.session_state["manual_table"] = _default_table()

left, right = st.columns([1.25, 1], gap="large")
with left, st.container(border=True):
    card_title("The example", "Six students: label +1 = passed, −1 = failed. Each weak learner is a simple rule.")
    pm = st.column_config.SelectboxColumn(options=[-1, 1], required=True)
    edited = st.data_editor(
        st.session_state["manual_table"], hide_index=True, width="stretch", key="manual_editor",
        disabled=["Sample", "Hours studied"],
        column_config={"True label y": pm, **{n: pm for n in LEARNER_NAMES}},
    )
    b1, b2 = st.columns(2)
    if b1.button("Reset to default example", icon=":material/restart_alt:"):
        st.session_state["manual_table"] = _default_table()
        st.session_state.pop("manual_editor", None)
        st.rerun()
    b2.caption("Change any label or prediction (−1 / +1) — everything below recalculates instantly.")

with right, st.container(border=True):
    card_title("Formulas used")
    st.latex(r"w_i^{(1)} = \frac{1}{N}")
    st.latex(r"\varepsilon_t = \sum_i w_i \cdot I\big(h_t(x_i) \neq y_i\big)")
    st.latex(r"\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}")
    st.latex(r"w_i^{new} = w_i^{old}\cdot e^{-\alpha_t\, y_i\, h_t(x_i)}")
    st.latex(r"w_i^{new} \leftarrow \frac{w_i^{new}}{\sum_j w_j^{new}}")
    st.latex(r"H(x) = \operatorname{sign}\Big(\sum_t \alpha_t\, h_t(x)\Big)")
    st.caption("y·h = +1 when the learner is right (weight shrinks by e^−α) and −1 when wrong (weight grows by e^α).")

y = edited["True label y"].to_numpy(dtype=int)
preds = {n: edited[n].to_numpy(dtype=int) for n in LEARNER_NAMES}
try:
    run = run_manual_adaboost(y, preds)
except ValueError as exc:
    st.error(f"Invalid example: {exc}")
    st.stop()

N = len(y)
st.markdown(f"### Step 1 — initial weights: every sample gets 1/N = 1/{N} = {1 / N:.4f}")

for t, rnd in enumerate(run.rounds, start=1):
    with st.container(border=True):
        card_title(f"Round {t} — {rnd.learner}")
        wrong = [f"x{i + 1}" for i in np.where(rnd.misclassified)[0]]
        c1, c2 = st.columns([1, 1.6])
        with c1:
            terms = " + ".join(f"{w:.4f}" for w in rnd.weights_before[rnd.misclassified]) or "0"
            st.markdown(f"**Step 2 — predictions.** Misclassified: **{', '.join(wrong) or 'none'}**")
            st.markdown("**Step 3 — weighted error** (sum of weights of the misclassified samples):")
            st.latex(rf"\varepsilon_{t} = {terms} = {rnd.error:.4f}")
            st.markdown("**Step 4 — learner weight:**")
            if rnd.status == "ok":
                st.latex(rf"\alpha_{t} = \tfrac12 \ln\frac{{1-{rnd.error:.4f}}}{{{rnd.error:.4f}}} = {rnd.alpha:.4f}")
                ex_r, ex_w = np.exp(-rnd.alpha), np.exp(rnd.alpha)
                st.markdown(f"**Step 5 — update:** correct samples × e^−α = × {ex_r:.4f}; "
                            f"wrong samples × e^α = × {ex_w:.4f}.")
                st.markdown(f"**Step 6 — normalise:** divide by the sum of new weights "
                            f"= {rnd.weights_unnormalized.sum():.4f}.")
            else:
                st.latex(rf"\alpha_{t} = {rnd.alpha:.4f}")
                callout(rnd.note, "warn")
        with c2:
            table = rnd.table(edited["Hours studied"].tolist(), y)
            st.dataframe(table.style.format({c: "{:.4f}" for c in ["Weight before", "exp(-α·y·h)", "Unnormalised",
                                                                  "Weight after"]})
                         .apply(lambda r: ["background-color:#FDE7EA" if r["Correct?"] == "No" else "" for _ in r],
                                axis=1),
                         hide_index=True, width="stretch")
            st.caption(f"Check: new weights sum to {rnd.weights_after.sum():.4f}. Misclassified rows are shaded — "
                       "their weights grow, so the next learner pays more attention to them.")

if run.stopped_early:
    callout("Boosting stopped early because a learner was perfect (ε = 0) or no better than chance (ε ≥ 0.5). "
            "Learners with ε ≥ 0.5 are left out of the final vote.", "warn")

with st.container(border=True):
    card_title("Step 7 — final weighted vote", "H(x) = sign(Σ αₜ·hₜ(x))")
    used = [r for r in run.rounds if r.status != "weak"]
    vote = pd.DataFrame({"Sample": edited["Sample"]})
    for r in used:
        vote[f"α·h ({r.learner.split(':')[0]})"] = np.round(r.alpha * r.predictions, 4)
    vote["Sum Σα·h"] = np.round(run.final_scores, 4)
    vote["Final H(x)"] = run.final_predictions
    vote["True y"] = y
    vote["Correct?"] = np.where(run.final_predictions == y, "Yes", "No")
    st.dataframe(vote, hide_index=True, width="stretch")
    singles = {r.learner: float(np.mean(r.predictions == y)) for r in used}
    st.markdown("**Accuracy:** " + " · ".join(f"{k.split(':')[0]} alone {v:.0%}" for k, v in singles.items())
                + f" → **combined ensemble {run.final_accuracy:.0%}**.")
    if run.final_accuracy > max(singles.values(), default=0):
        st.success("The weighted vote beats every individual weak learner — that is the core idea of boosting.")
    st.plotly_chart(viz.manual_weights_fig(run.rounds), width="stretch", config=PLOTLY_CONFIG)

with st.expander("Edge cases handled"):
    st.markdown("""
- **ε = 0 (perfect learner):** α = ½·ln(1/0) is infinite. We clip ε to 10⁻¹⁰ to show a finite (very large) α and stop
  boosting — a perfect learner needs no help.
- **ε = 0.5:** α = 0 — the learner is a coin flip and has no say.
- **ε > 0.5:** α < 0 — the learner is worse than random. AdaBoost discards it and stops (as scikit-learn does).
- **Invalid input:** labels/predictions other than −1/+1, or missing predictions, are rejected with an error message.
""")
