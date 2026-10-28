"""Single source of truth for the app's pages (used by app.py, sidebar and quick search)."""

# (file path, title, material icon, section)
PAGES = [
    ("views/dashboard.py", "Dashboard", ":material/space_dashboard:", "main"),
    ("views/digit_predictor.py", "Digit Predictor", ":material/draw:", "main"),
    ("views/dataset_explorer.py", "Dataset Explorer", ":material/dataset:", "main"),
    ("views/model_training.py", "Model Training", ":material/model_training:", "main"),
    ("views/model_evaluation.py", "Model Evaluation", ":material/fact_check:", "main"),
    ("views/algorithm_comparison.py", "Algorithm Comparison", ":material/compare_arrows:", "main"),
    ("views/visualizations.py", "Visualizations", ":material/insights:", "main"),
    ("views/manual_adaboost.py", "Manual AdaBoost", ":material/calculate:", "learn"),
    ("views/model_explainability.py", "How the Model Thinks", ":material/psychology:", "learn"),
    ("views/rubric_checklist.py", "Rubric Checklist", ":material/checklist:", "learn"),
    ("views/about_project.py", "About Project", ":material/description:", "learn"),
]

PATH_BY_TITLE = {title: path for path, title, _icon, _sec in PAGES}
