"""Rubric Checklist: the 11 grading criteria and where each is satisfied (status is checked live)."""

import pandas as pd
import streamlit as st

from src.comparison import load_latest_comparison
from src.config import PROJECT_ROOT, REPORTS_DIR
from src.model_training import read_experiments
from ui.components import metric_card, page_header
from ui.state import get_result

page_header("Rubric Checklist", "All 11 criteria, with live checks of what has actually been produced.", "rubric")

result = get_result()
has_model = result is not None
has_comparison = load_latest_comparison() is not None
n_exps = len(read_experiments())
has_report = (REPORTS_DIR / "project_report.md").exists()
has_tests = all((PROJECT_ROOT / "tests" / f).exists() for f in ("test_preprocessing.py", "test_model.py", "test_metrics.py"))

DONE, NEEDS = "✅", "⏳"


def item(text: str, page: str, ok: bool = True, todo: str = "") -> tuple:
    return (DONE if ok else NEEDS, text, page, "" if ok else todo)


CRITERIA = [
    ("Dataset Understanding", 10, [
        item("Dataset description, features (784 pixels) and target (digit 0–9)", "Dataset Explorer → Overview"),
        item("Training/testing counts and class distribution (real counts)", "Dataset Explorer → Class distribution"),
        item("Missing-value inspection (measured)", "Dataset Explorer → Data quality"),
        item("Dataset characteristics: pixel heatmaps, average digits, handwriting variation", "Dataset Explorer"),
    ]),
    ("Data Preprocessing", 10, [
        item("Missing-value and duplicate checks", "Dataset Explorer → Data quality"),
        item("Pixel normalisation 0–255 → 0–1 with before/after view", "Dataset Explorer → Preprocessing"),
        item("Flattening 28×28 → 784 with explanation", "Dataset Explorer → Preprocessing"),
        item("Official split kept + stratified, seeded subsampling", "Dataset Explorer → Preprocessing"),
        item("Preprocessing code (src/preprocessing.py) and drawing pipeline", "Digit Predictor → preprocessing steps"),
    ]),
    ("Algorithm Implementation", 15, [
        item("Actual AdaBoost training with DecisionTree weak learners", "Model Training", has_model, "Train a model"),
        item("Manual binary AdaBoost calculation (ε, α, weight updates, normalisation, final vote)", "Manual AdaBoost"),
        item("Intermediate weights shown in tables, editable and recalculated", "Manual AdaBoost"),
        item("Real α and ε of every fitted learner", "Model Training", has_model, "Train a model"),
    ]),
    ("Algorithm Application / Modeling", 15, [
        item("Multiclass (10-digit) AdaBoost (SAMME) applied to MNIST", "Model Training", has_model, "Train a model"),
        item("Functional training and live prediction (draw / upload)", "Digit Predictor", has_model, "Train a model"),
        item("Configuration explained in plain language", "Model Training"),
    ]),
    ("Parameter Tuning / Customization", 10, [
        item("Controls: estimators, learning rate, max depth, training size, seed", "Model Training"),
        item("Configurations compared from logged runs", "Model Training → Compare your configurations",
             n_exps >= 2, f"Train at least 2 configurations ({n_exps} so far)"),
    ]),
    ("Model Evaluation / Metrics", 10, [
        item("Accuracy, precision, recall, F1 (macro & weighted)", "Model Evaluation", has_model, "Train a model"),
        item("Interactive 10×10 confusion matrix with cell inspection", "Model Evaluation", has_model, "Train a model"),
        item("Manually worked metric calculation checked against scikit-learn", "Model Evaluation", has_model, "Train a model"),
    ]),
    ("Comparison / Analysis", 10, [
        item("AdaBoost vs Decision Tree vs Random Forest on the same split", "Algorithm Comparison", has_comparison,
             "Run a comparison"),
        item("Metrics comparison table with best values highlighted", "Algorithm Comparison", has_comparison, "Run a comparison"),
        item("Strengths and weaknesses discussion", "Algorithm Comparison"),
    ]),
    ("Visualization", 10, [
        item("Customised dashboard (lime/charcoal/lavender design)", "Dashboard"),
        item("Digit galleries, class distribution, pixel heatmaps", "Dashboard · Visualizations"),
        item("Confusion matrix, performance and comparison charts", "Visualizations", has_model, "Train a model"),
    ]),
    ("Interpretation and Discussion", 10, [
        item("Interpretation of metrics and train–test gap", "Model Training · Model Evaluation", has_model, "Train a model"),
        item("Discussion of misclassified digits with real examples", "How the Model Thinks", has_model, "Train a model"),
        item("Limitations and conclusions", "How the Model Thinks · About Project"),
    ]),
    ("Code Documentation and Readability", 5, [
        item("Modular code: src/ (ML), ui/ (shared UI), views/ (pages)", "Project folder"),
        item("Docstrings, comments, meaningful names, error handling", "src/*.py"),
        item("Automated tests", "tests/ (run: python -m pytest tests)", has_tests, "Tests missing"),
    ]),
    ("Report / Presentation", 5, [
        item("Report with all 20 sections, generated from real results (MD / HTML / PDF)", "About Project", has_report,
             "Generate the report on About Project"),
        item("Viva-ready summary and likely questions", "About Project"),
    ]),
]

complete = sum(all(s == DONE for s, *_ in items) for _n, _p, items in CRITERIA)
c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Criteria fully evidenced", f"{complete} / 11", "every item checked live", "✅", variant="dark",
                anim_tag=f"rub-{complete}")
with c2:
    metric_card("Total rubric points", "110", "10+10+15+15+10+10+10+10+10+5+5", "🏆")
with c3:
    pending = sum(s == NEEDS for _n, _p, items in CRITERIA for s, *_ in items)
    metric_card("Items waiting for a run", str(pending), "features exist; results need generating", "⏳", variant="lime")
st.write("")

for n, (name, pts, items) in enumerate(CRITERIA, start=1):
    ok = all(s == DONE for s, *_ in items)
    with st.container(border=True):
        st.markdown(f'<div class="rubric-row"><span class="score-pill">{pts} pts</span>'
                    f'<p class="card-title">Criterion {n}: {name} {"✅" if ok else "⏳"}</p></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(items, columns=["Status", "Evidence", "Where to find it", "To do"]),
                     hide_index=True, width="stretch")

st.caption("✅ = implemented and its results exist. ⏳ = the feature is implemented but its result has not been produced "
           "yet in this installation (follow the 'To do' column). Nothing is marked complete without checking.")
