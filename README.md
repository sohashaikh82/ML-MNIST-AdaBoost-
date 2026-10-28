# MNIST AI LAB — Classify Handwritten Digits (MNIST) Using AdaBoost

An interactive machine-learning mini-project: train **AdaBoost** (with decision-tree weak learners) on the real
MNIST dataset, draw digits for live prediction, explore the data, evaluate the model, compare it with a Decision
Tree and a Random Forest, and follow a fully worked **manual AdaBoost calculation** — all in a Streamlit dashboard.

Every metric, chart and table is computed from real data or a real model run. Nothing is hard-coded.

---

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Live metric cards (dataset size, accuracy vs. single tree, training time, classes), class distribution, sample gallery, actual-vs-predicted, confusion-matrix preview, comparison and recent predictions |
| **Digit Predictor** | Draw on a black canvas (pen / eraser / thickness / clear) or upload an image → MNIST-style preprocessing (shown step by step) → predicted digit, weighted vote share and raw scores per digit, latency, explanation, history |
| **Dataset Explorer** | What MNIST is, counts, class balance, browse by digit, pixel heatmaps, 28×28 matrices, average digits, handwriting variation, data-quality checks (missing values, duplicates), preprocessing before/after |
| **Model Training** | Presets + controls for estimators, learning rate, tree depth, sample size, seed; real training with elapsed time; staged accuracy curve; real α/ε per learner; experiment log to compare configurations; step-by-step algorithm explanation |
| **Model Evaluation** | Accuracy, precision, recall, F1 (macro & weighted), clickable 10×10 confusion matrix, cell inspector, top confusions, classification report, manual metric calculation verified against scikit-learn, timing |
| **Algorithm Comparison** | Decision Tree vs AdaBoost vs Random Forest on the identical split; table with best values highlighted; charts; analysis; results cached |
| **Visualizations** | Gallery of every chart (data, model, results, comparison, manual AdaBoost) |
| **Manual AdaBoost** | Six-sample binary example: initial weights, ε, α, weight update, normalisation, final vote — editable and recalculated live, with edge cases (ε = 0, ε ≥ 0.5, invalid input) |
| **How the Model Thinks** | Weak learners, boosting vs bagging, feature-importance map, first split of each tree, real misclassified examples with scores, effect of parameters, limitations |
| **Rubric Checklist** | All 11 grading criteria with live status checks |
| **About Project** | Viva summary + likely questions; generates the project report (Markdown, HTML, PDF) from your actual results |

## Results included in this folder (measured, not invented)

Trained on a stratified 10,000-image subset, tested on the full official 10,000-image test set (seed 42):

| Configuration | Test accuracy | Single-tree baseline | Training time* |
|---|---|---|---|
| 50 stumps (depth 1) | 63.69% | 19.31% | 7.3 s |
| 30 trees, depth 4, 5,000 images | 87.46% | 64.08% | 7.6 s |
| **50 trees, depth 6 (default, active model)** | **93.55%** | 72.64% | 41.7 s |
| 50 trees, depth 8 | 94.85% | 80.45% | 51.2 s |
| 100 trees, depth 10 | 96.02% | 82.38% | 120.4 s |

Comparison on the same split: Decision Tree (depth 12) 82.32% · **AdaBoost 93.55%** · Random Forest (100 trees) 95.13%.

\*Measured on a 2-core cloud machine; your laptop's times will differ.

---

## Technology stack

Python · scikit-learn · NumPy · Pandas · SciPy · Streamlit · Plotly · Matplotlib · Seaborn · Pillow ·
streamlit-drawable-canvas · Joblib · ReportLab · pytest

## Dataset

MNIST: 70,000 grayscale 28×28 images of handwritten digits — 60,000 for training and 10,000 for testing.
A copy is included at `data/mnist.npz` (11 MB) so the app works **offline**. If that file is missing, the app
downloads MNIST automatically (Keras copy → GitHub mirror → OpenML) and caches it.

---

## Installation (Windows, step by step)

You need **Python 3.11 or newer** (check with `python --version`). In VS Code, open the
`mnist-adaboost-project` folder, then open a terminal (**Terminal → New Terminal**) and run:

**1. Create a virtual environment** (a private folder for this project's packages)
```
python -m venv .venv
```

**2. Activate it**
```
.venv\Scripts\activate
```
You should now see `(.venv)` at the start of the terminal line.
*(macOS/Linux: `source .venv/bin/activate`)*

> If PowerShell says *"running scripts is disabled"*, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**3. Install the requirements** (takes a few minutes the first time)
```
pip install -r requirements.txt
```

## Running the application

```
streamlit run app.py
```
Your browser opens at <http://localhost:8501>. Stop the app with **Ctrl + C** in the terminal.

The main entry file is **`app.py`** in the project folder.

## How to train the model

1. Open **Model Training** in the sidebar.
2. Pick a preset (*Quick demo* ≈ 10 s, *Recommended* ≈ 40 s) or set the sliders yourself.
3. Press **Train Model**. You'll see the real elapsed time, then accuracy, charts and the configuration.
   The model is saved in `models/` and becomes the active model on every page.
   Training the same settings again reuses the saved model (tick *Retrain anyway* to force it).

A trained default model is already included, so every page works immediately.

## How to draw and predict a digit

1. Open **Digit Predictor**.
2. Draw one digit large and centred on the black canvas (switch to **Erase** to fix mistakes,
   **Clear canvas** to start over). Or choose **Upload image**.
3. Press **Predict Digit** → see the predicted digit, scores, the processed 28×28 image and an explanation.
4. Press **Try Another Digit** for a fresh canvas. Your predictions are listed under *Prediction history*.

## Running the automated tests

```
python -m pytest tests
```
50 tests cover dataset loading/validation, preprocessing, input dimensions, blank and invalid images, training on
a small sample, prediction output, save/load, metrics, confusion matrix, the manual AdaBoost formulas and invalid
parameters.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the virtual environment first (`.venv\Scripts\activate`). |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated environment. |
| Sidebar says **No model yet** although models exist | The saved models were made with a different scikit-learn version. Train once on Model Training (the default takes ~40 s). |
| Canvas doesn't appear | Make sure `streamlit-drawable-canvas` installed; use **Upload image** meanwhile. |
| Wrong predictions on drawings | Draw bigger, thicker and centred. Mouse drawings differ from MNIST's pen scans — this is a known limitation. |
| Dataset error | Keep `data/mnist.npz` in place, or connect to the internet so it can be downloaded. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Training is slow | Use a preset, a smaller sample size or a smaller max depth. AdaBoost trains trees one after another, so it can't use multiple CPU cores. |

## Project files

```
mnist-adaboost-project/
├── app.py                    Entry point: page setup, navigation, sidebar
├── requirements.txt          Python packages
├── README.md                 This file
├── .streamlit/config.toml    Theme colours and server settings
├── assets/                   styles.css (dashboard design) and logo.png
├── data/mnist.npz            Local copy of MNIST (official 60k/10k split)
├── src/                      Machine-learning code (no Streamlit — testable on its own)
│   ├── config.py             Paths and default hyper-parameters
│   ├── data_loader.py        Load MNIST (cache + online fallbacks) and validate it
│   ├── preprocessing.py      Normalise, flatten, subsample; drawing/upload → 28×28 pipeline
│   ├── adaboost_model.py     AdaBoost configuration and model builder (version-compatible)
│   ├── model_training.py     Training, staged accuracy, baseline, save/load, experiment log
│   ├── evaluation.py         Metrics, confusion matrix, manual per-class calculation
│   ├── comparison.py         Decision Tree vs AdaBoost vs Random Forest
│   ├── manual_adaboost.py    Worked binary AdaBoost example with edge cases
│   ├── visualization.py      Plotly chart builders
│   └── report.py             Report generator (Markdown / HTML / PDF + figures)
├── ui/                       Shared Streamlit code: nav.py, state.py (caching), components.py
├── views/                    One file per page (11 pages)
├── models/                   Saved models, evaluations, comparison results, experiments.json
├── reports/                  Generated report (project_report.md/.html/.pdf) and figures/
└── tests/                    pytest suite
```

> Note: pages live in `views/` (not `pages/`) because the app uses Streamlit's `st.navigation` with a custom
> sidebar; a folder named `pages/` would make Streamlit add a second, automatic menu.

## Known limitations

- The **weighted vote share** shown with a prediction (fraction of the trees' α-weighted votes per digit) is *not* a calibrated probability.
- scikit-learn does not report progress *during* `fit`, so training shows real elapsed time instead of a percentage bar.
- Streamlit has no reliable way to switch the app between light and dark themes from inside a page, so the custom
  dashboard styling is light-only (no theme toggle was added rather than shipping a half-working one).
- Animations are CSS-only and play once per session; they are disabled automatically if your system asks for reduced motion.
