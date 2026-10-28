"""About Project: viva summary and downloadable report generated from real results."""

import re

import streamlit as st

from src.comparison import load_latest_comparison
from src.config import REPORTS_DIR
from src.model_training import read_experiments
from src.report import build_context, generate_report
from ui.components import callout, card_title, page_header
from ui.state import get_evaluation, get_result, get_validation_report, require_data

page_header("About Project", "Viva summary and a report generated from your actual results.", "about")
data = require_data()
result = get_result()
ev = get_evaluation(result) if result else None
comp = load_latest_comparison()

tab_viva, tab_report, tab_about = st.tabs(["🎤 Viva summary", "📄 Project report", "ℹ️ About & architecture"])

with tab_viva:
    c1, c2 = st.columns([1.2, 1])
    with c1, st.container(border=True):
        card_title("Explain the project in 60 seconds")
        acc = f"{ev['accuracy']:.1%}" if ev else "<train a model>"
        base = f"{result.baseline_test_accuracy:.1%}" if result else "<baseline>"
        n_est = result.n_estimators_fitted if result else "N"
        depth = result.config.max_depth if result else "d"
        st.markdown(f"""
1. **Problem** — recognise handwritten digits 0–9 from 28 × 28 grayscale images (MNIST, 70,000 images).
2. **Data** — official split of 60,000 training and 10,000 test images; checked for missing values and duplicates;
   pixels normalised to 0–1 and flattened to 784 features.
3. **Algorithm** — AdaBoost: train small decision trees one after another; after each tree, give more weight to the
   images it got wrong; combine all trees with a weighted vote (α).
4. **Result** — {n_est} trees of depth {depth} reach **{acc}** test accuracy vs **{base}** for one tree alone.
5. **Analysis** — compared with a Decision Tree and a Random Forest on the same split; inspected the confusion
   matrix (e.g. 4↔9, 3↔5 confusions); worked one AdaBoost round by hand.
6. **Demo** — draw a digit; the app preprocesses it like MNIST and shows the predicted digit and class scores.
""")
    with c2, st.container(border=True):
        card_title("Likely viva questions")
        with st.expander("Why AdaBoost and not a neural network?"):
            st.write("The project's goal is to study boosting. AdaBoost is interpretable (we can see each learner's α, "
                     "error and splits) and trains on a CPU in under a minute. CNNs are more accurate on images but "
                     "harder to explain.")
        with st.expander("What is a weak learner?"):
            st.write("A model only slightly better than random guessing — here a shallow decision tree.")
        with st.expander("How does AdaBoost update weights?"):
            st.write("Misclassified samples are multiplied by e^α (binary form: w·e^(−α·y·h)), then all weights are "
                     "normalised to sum to 1, so the next learner focuses on hard samples.")
        with st.expander("What happens if a learner's error is 0.5 or more?"):
            st.write("α becomes 0 or negative — the learner is no better than chance — so it is discarded and boosting "
                     "stops. For K classes (SAMME) the threshold is ε < 1 − 1/K = 0.9.")
        with st.expander("Why flatten the images?"):
            st.write("scikit-learn needs a 2-D table (samples × features); every pixel becomes one feature.")
        with st.expander("Macro vs weighted average?"):
            st.write("Macro = plain mean over the 10 digits; weighted = mean weighted by each digit's number of test "
                     "images. MNIST is nearly balanced, so they are close.")
        with st.expander("Is the vote share a probability?"):
            st.write("No. It is the fraction of the trees' α-weighted votes each digit received — useful for comparing "
                     "digits, but not calibrated (it does not mean the model is right that often).")
        with st.expander("Boosting vs bagging?"):
            st.write("Boosting builds learners sequentially, each fixing previous errors (reduces bias). Bagging "
                     "(Random Forest) builds independent learners on bootstrap samples and averages them (reduces variance).")

with tab_report:
    with st.container(border=True):
        card_title("Generate the project report", "Built from the active model, its evaluation, the latest comparison "
                                                  "and your experiment log. Missing results are labelled, never invented.")
        c1, c2 = st.columns(2)
        author = c1.text_input("Student name(s) for the title page", placeholder="e.g. Your Name")
        institute = c2.text_input("Institute / department", placeholder="e.g. Department of CSBS")
        if not result:
            callout("No trained model yet — the report will mark the results sections as pending.", "warn")
        if not comp:
            callout("No algorithm comparison yet — that section will be marked as pending.", "warn")
        if st.button("Generate report", type="primary", icon=":material/description:"):
            with st.spinner("Drawing figures and writing the report…"):
                ctx = build_context(data, get_validation_report(), result, ev, comp[1] if comp else None,
                                    read_experiments(), author, institute)
                st.session_state["report_paths"] = generate_report(ctx, REPORTS_DIR)
            st.toast("Report generated", icon="📄")

    paths = st.session_state.get("report_paths")
    if paths is None and (REPORTS_DIR / "project_report.md").exists():
        paths = {k: REPORTS_DIR / f"project_report.{k}" for k in ("md", "html", "pdf")
                 if (REPORTS_DIR / f"project_report.{k}").exists()}
        st.caption("Showing the most recently generated report (press Generate to refresh it with current results).")
    if paths:
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download Markdown", paths["md"].read_bytes(), "MNIST_AdaBoost_Report.md", "text/markdown",
                           icon=":material/download:", width="stretch")
        d2.download_button("Download HTML (print to PDF)", paths["html"].read_bytes(), "MNIST_AdaBoost_Report.html",
                           "text/html", icon=":material/download:", width="stretch")
        if "pdf" in paths:
            d3.download_button("Download PDF", paths["pdf"].read_bytes(), "MNIST_AdaBoost_Report.pdf", "application/pdf",
                               icon=":material/picture_as_pdf:", width="stretch")
        with st.container(border=True):
            md = paths["md"].read_text(encoding="utf-8")
            # Render the report with its figures inline (Markdown images can't load local files directly)
            parts = re.split(r"!\[(.*?)\]\((figures/[^)]+)\)", md)
            for i in range(0, len(parts), 3):
                st.markdown(parts[i])
                if i + 2 < len(parts):
                    st.image(str(REPORTS_DIR / parts[i + 2]), caption=parts[i + 1], width=640)

with tab_about:
    with st.container(border=True):
        card_title("About MNIST AI LAB")
        st.markdown("""
An interactive college mini-project for **classifying handwritten digits with AdaBoost**. Everything shown is
computed from the real MNIST dataset and real model runs — no hard-coded results.

**Architecture**
```
app.py                 → page setup + navigation (st.navigation / st.page_link)
ui/                    → shared Streamlit code: styling, cached state, components
views/                 → one file per page (11 pages)
src/                   → pure-Python ML code (no Streamlit): data, preprocessing,
                         model, training, evaluation, comparison, manual AdaBoost, report
models/                → saved models, evaluations, comparison results, experiment log
reports/               → generated report (Markdown, HTML, PDF) and figures
tests/                 → pytest suite
```
**Tech:** Python · scikit-learn · NumPy · Pandas · Streamlit · Plotly · Matplotlib/Seaborn · Pillow · SciPy ·
streamlit-drawable-canvas · Joblib · ReportLab · pytest
""")
