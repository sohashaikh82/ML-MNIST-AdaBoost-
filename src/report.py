"""Project report generation (Markdown, HTML and PDF) from real experiment results.

Figures are drawn with Matplotlib/Seaborn so the report works offline and
without a browser. Any section that lacks results (e.g. no comparison run yet)
says so explicitly instead of inventing numbers.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PURPLE, LAVENDER, LIME, CHARCOAL = "#7C5CE6", "#C9BEFA", "#9CC33A", "#1E1F24"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(ctx: dict, out_dir: Path) -> dict:
    """Save report figures as PNG; returns {name: relative path}."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    data = ctx["data"]
    # sample digits
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.3))
    for d in range(10):
        idx = np.where(data.y_train == d)[0][:2]
        for r in range(2):
            axes[r, d].imshow(data.x_train[idx[r]], cmap="gray")
            axes[r, d].axis("off")
        axes[0, d].set_title(str(d))
    fig.tight_layout()
    fig.savefig(fig_dir / "samples.png", dpi=150)
    plt.close(fig)
    paths["samples"] = "figures/samples.png"

    # class distribution
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(10)
    ax.bar(x - 0.2, np.bincount(data.y_train, minlength=10), 0.4, label="Train", color=PURPLE)
    ax.bar(x + 0.2, np.bincount(data.y_test, minlength=10), 0.4, label="Test", color=LIME)
    ax.set_xticks(x)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Images")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    paths["class_distribution"] = "figures/class_distribution.png"

    result, ev = ctx.get("result"), ctx.get("evaluation")
    if result is not None and ev is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(ev["confusion_matrix"], annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                    annot_kws={"size": 7})
        ax.set_xlabel("Predicted Digit")
        ax.set_ylabel("Actual Digit")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        paths["confusion_matrix"] = "figures/confusion_matrix.png"

        fig, ax = plt.subplots(figsize=(7, 3))
        rounds = np.arange(1, len(result.staged_test_accuracy) + 1)
        ax.plot(rounds, result.staged_train_accuracy, ":", color=LIME, label="Training subset")
        ax.plot(rounds, result.staged_test_accuracy, color=PURPLE, lw=2, label="Test set")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "staged_accuracy.png", dpi=150)
        plt.close(fig)
        paths["staged_accuracy"] = "figures/staged_accuracy.png"

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.imshow(result.model.feature_importances_.reshape(28, 28), cmap="Purples")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "feature_importance.png", dpi=150)
        plt.close(fig)
        paths["feature_importance"] = "figures/feature_importance.png"

    comp = ctx.get("comparison")
    if comp is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        colors = [LAVENDER, PURPLE, LIME]
        axes[0].bar(comp["Model"], comp["Test accuracy"], color=colors)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Test accuracy")
        axes[1].bar(comp["Model"], comp["Training time (s)"], color=colors)
        axes[1].set_title("Training time (s)")
        for ax in axes:
            ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison.png", dpi=150)
        plt.close(fig)
        paths["comparison"] = "figures/comparison.png"
    return paths


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _pct(v) -> str:
    return f"{v:.2%}"


def build_markdown(ctx: dict, figures: dict | None = None) -> str:
    figures = figures or {}
    data, val = ctx["data"], ctx["validation"]
    result, ev, comp, manual = ctx.get("result"), ctx.get("evaluation"), ctx.get("comparison"), ctx.get("manual")
    author = ctx.get("author") or "Student Name"
    institute = ctx.get("institute") or "Institute Name"
    L: list[str] = []
    add = L.append

    def fig(name, caption):
        if name in figures:
            add(f"![{caption}]({figures[name]})\n\n*Figure: {caption}*\n")

    add("# Classify Handwritten Digits (MNIST) Using AdaBoost\n")
    add(f"**Machine Learning Mini-Project Report**  \n**Submitted by:** {author}  \n**Institute:** {institute}  \n"
        f"**Date:** {date.today().strftime('%d %B %Y')}\n")
    add("---\n")

    add("## 1. Abstract\n")
    if result and ev:
        add(f"This project builds an interactive system that recognises handwritten digits using the AdaBoost ensemble "
            f"algorithm with decision-tree weak learners. Trained on {result.config.train_size:,} MNIST images, "
            f"{result.n_estimators_fitted} boosted trees of depth {result.config.max_depth} reached "
            f"**{_pct(ev['accuracy'])} accuracy** and a macro F1-score of {_pct(ev['f1_macro'])} on the official "
            f"10,000-image test set, compared with {_pct(result.baseline_test_accuracy)} for a single tree of the same "
            "depth. A Streamlit dashboard lets users draw digits, tune hyper-parameters, inspect the data, compare "
            "AdaBoost with Decision Tree and Random Forest, and follow a manual AdaBoost calculation.\n")
    else:
        add("*(Results pending: train a model in the application and regenerate this report.)*\n")

    add("## 2. Introduction\n")
    add("Handwritten digit recognition is used for postal codes, bank cheques and form digitisation. It is a standard "
        "benchmark for classification algorithms because handwriting varies widely between people. Ensemble methods "
        "combine many simple models into a stronger one; AdaBoost is the classic boosting algorithm.\n")
    add("## 3. Problem Statement\n")
    add("Given a 28 × 28 grayscale image of a single handwritten digit, predict which digit (0–9) it shows, using "
        "AdaBoost as the primary classifier, and evaluate how well it generalises to handwriting it has never seen.\n")
    add("## 4. Objectives\n")
    for o in ["Understand and validate the MNIST dataset.", "Build a reproducible preprocessing pipeline.",
              "Implement and tune multiclass AdaBoost with decision-tree weak learners.",
              "Demonstrate AdaBoost's weight updates through a manual calculation.",
              "Evaluate with accuracy, precision, recall, F1 and a confusion matrix.",
              "Compare AdaBoost with Decision Tree and Random Forest under identical conditions.",
              "Deliver an interactive dashboard with live digit prediction."]:
        add(f"- {o}")
    add("")

    add("## 5. Dataset Description\n")
    tr, te = val["train_class_counts"], val["test_class_counts"]
    add(f"MNIST contains {data.n_train + data.n_test:,} grayscale images of handwritten digits (28 × 28 = 784 pixels, "
        f"values 0–255) with the official split of **{data.n_train:,} training** and **{data.n_test:,} test** images. "
        "**Features:** the 784 pixel intensities. **Target:** the digit label 0–9 (10 classes).\n")
    add("| Digit | " + " | ".join(map(str, range(10))) + " |")
    add("|---|" + "---|" * 10)
    add("| Train | " + " | ".join(f"{c:,}" for c in tr) + " |")
    add("| Test | " + " | ".join(f"{c:,}" for c in te) + " |\n")
    add(f"Validation (computed): pixel range {val['pixel_min']}–{val['pixel_max']}; missing values: "
        f"{val['missing_train_pixels'] + val['missing_test_pixels'] + val['missing_labels']}; exact duplicate training "
        f"images: {val['duplicate_train_images']}; images present in both train and test: {val['train_test_overlap']}.\n")
    fig("samples", "Sample training images (two per digit)")
    fig("class_distribution", "Class distribution in the training and test sets")

    add("## 6. Tools and Technologies\n")
    add("Python · scikit-learn (AdaBoostClassifier, DecisionTreeClassifier, RandomForestClassifier, metrics) · NumPy · "
        "Pandas · Streamlit (web app, session state, navigation) · Plotly (interactive charts) · Matplotlib/Seaborn "
        "(report figures) · Pillow & SciPy (image preprocessing) · streamlit-drawable-canvas (drawing input) · Joblib "
        "(model persistence) · pytest (automated tests).\n")

    add("## 7. Data Preprocessing\n")
    for s in ["**Loading** the official 60,000/10,000 split (local cache with online fallbacks).",
              "**Validation**: shapes, label set, pixel range, missing values (none found) and duplicate checks.",
              "**Normalisation**: pixel / 255 → values in [0, 1]. A fixed transform, so no information leaks from the test set.",
              "**Flattening**: each 28 × 28 image → a 784-feature vector, because scikit-learn expects a 2-D samples × features table.",
              "**Stratified subsampling** of the training set (keeps each digit's proportion) to keep AdaBoost fast.",
              "**Fixed random seed** for reproducible subsets and models.",
              "**User drawings**: grayscale → invert if needed → crop to the digit → resize longest side to 20 px "
              "(aspect kept) → centre in 28 × 28 by centre of mass → normalise."]:
        add(f"1. {s}")
    add("")

    add("## 8. AdaBoost Theory\n")
    add("AdaBoost (Adaptive Boosting) trains weak learners sequentially. Each training sample has a weight; after each "
        "round, misclassified samples receive more weight so the next learner focuses on them. Each learner gets a vote "
        "α based on its weighted error ε, and the final prediction is the weighted vote. For K classes scikit-learn uses "
        "SAMME: α = η·[ln((1−ε)/ε) + ln(K−1)], and misclassified sample weights are multiplied by e^α.\n")

    add("## 9. Manual AdaBoost Calculation\n")
    if manual is not None:
        add("Binary example with six samples (labels −1/+1), N = 6, initial weights 1/6 ≈ 0.1667.\n")
        add("| Round | Learner | ε | α | Weights after (x1…x6) |")
        add("|---|---|---|---|---|")
        for i, r in enumerate(manual.rounds, 1):
            add(f"| {i} | {r.learner} | {r.error:.4f} | {r.alpha:.4f} | {', '.join(f'{w:.4f}' for w in r.weights_after)} |")
        add("")
        r1 = manual.rounds[0]
        add(f"Round 1 worked: ε₁ = 1/6 = {r1.error:.4f}; α₁ = ½·ln((1−{r1.error:.4f})/{r1.error:.4f}) = ½·ln 5 = "
            f"{r1.alpha:.4f}. Correct samples are multiplied by e^−α = {np.exp(-r1.alpha):.4f}, the wrong one by "
            f"e^α = {np.exp(r1.alpha):.4f}; after normalising, correct samples have weight 0.1000 and the misclassified "
            "sample 0.5000.\n")
        add(f"Final vote H(x) = sign(Σ αₜhₜ(x)) gives scores {', '.join(f'{s:.3f}' for s in manual.final_scores)} → "
            f"ensemble accuracy **{manual.final_accuracy:.0%}**, although each weak learner alone makes mistakes.\n")
        add("*This binary example is educational and separate from the multiclass MNIST model.*\n")

    add("## 10. System Architecture\n")
    add("```\nMNIST data ──► src/data_loader (load + validate)\n        │\n        ▼\nsrc/preprocessing "
        "(normalise, flatten, subsample, user-image pipeline)\n        │\n        ▼\nsrc/model_training ──► "
        "src/adaboost_model (AdaBoost + tree weak learners) ──► models/*.joblib\n        │\n        ▼\nsrc/evaluation "
        "· src/comparison · src/manual_adaboost · src/report\n        │\n        ▼\nui/ (state, styling, components) ──► "
        "views/ (11 Streamlit pages) ──► app.py (navigation)\n```\n")

    add("## 11. Implementation Methodology\n")
    add("The ML code (`src/`) is independent of the user interface and covered by automated tests. The Streamlit app "
        "caches the dataset, saves every trained model with its configuration and measured metrics, reuses a saved "
        "model when the same configuration is requested again, and logs every experiment for comparison.\n")

    add("## 12. Hyperparameter Configuration\n")
    if result:
        c = result.config
        add("| Parameter | Value | Meaning |\n|---|---|---|")
        add(f"| n_estimators | {c.n_estimators} ({result.n_estimators_fitted} fitted) | number of boosting rounds |")
        add(f"| learning_rate | {c.learning_rate} | shrinks each learner's vote |")
        add(f"| max_depth (weak learner) | {c.max_depth} | strength of each tree |")
        add(f"| training images | {c.train_size:,} | stratified subset |")
        add(f"| random_state | {c.random_state} | reproducibility |\n")
    exps = ctx.get("experiments") or []
    if len(exps) > 1:
        add("Logged experiments (measured):\n")
        add("| # | Estimators | Depth | LR | Train size | Test acc. | Time (s) |\n|---|---|---|---|---|---|---|")
        for i, e in enumerate(exps, 1):
            add(f"| {i} | {e['n_estimators']} | {e['max_depth']} | {e['learning_rate']} | {e['train_size']:,} | "
                f"{_pct(e['test_accuracy'])} | {e['train_time_s']:.1f} |")
        add("")

    add("## 13. Model Evaluation\n")
    if result and ev:
        add("| Metric | Macro | Weighted |\n|---|---|---|")
        add(f"| Precision | {_pct(ev['precision_macro'])} | {_pct(ev['precision_weighted'])} |")
        add(f"| Recall | {_pct(ev['recall_macro'])} | {_pct(ev['recall_weighted'])} |")
        add(f"| F1-score | {_pct(ev['f1_macro'])} | {_pct(ev['f1_weighted'])} |\n")
        add(f"**Accuracy:** {_pct(ev['accuracy'])} ({ev['n_correct']:,}/{ev['n_test']:,}). **Training time:** "
            f"{result.train_time_s:.1f} s. **Inference:** {ev['inference_ms_per_image']:.4f} ms per image.\n")
        rep = ev["report_df"].iloc[:10]
        add("| Digit | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
        for name, row in rep.iterrows():
            add(f"| {name.replace('Digit ', '')} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | "
                f"{int(row['support'])} |")
        add("")
        fig("confusion_matrix", "Confusion matrix on the 10,000 test images")
        fig("staged_accuracy", "Accuracy after each boosting round")
    else:
        add("*(No trained model — results pending.)*\n")

    add("## 14. Algorithm Comparison\n")
    if comp is not None:
        add("All models used the same stratified training subset and the full test set.\n")
        add("| Model | Accuracy | Macro P | Macro R | Macro F1 | Train (s) | Inference (s) |\n|---|---|---|---|---|---|---|")
        for _, r in comp.iterrows():
            add(f"| {r['Model']} | {_pct(r['Test accuracy'])} | {_pct(r['Macro precision'])} | {_pct(r['Macro recall'])} | "
                f"{_pct(r['Macro F1'])} | {r['Training time (s)']:.2f} | {r['Inference time (s)']:.3f} |")
        add("")
        fig("comparison", "Model comparison")
        add("Results depend on the training subset, hyper-parameters, random seed and hardware.\n")
    else:
        add("*(Comparison not yet run — use the Algorithm Comparison page, then regenerate.)*\n")

    add("## 15. Visualizations\n")
    add("The dashboard provides: digit galleries, class-distribution charts, pixel-intensity heatmaps, average digit "
        "images, an interactive confusion matrix with cell inspection, staged-accuracy and α/ε charts, a pixel "
        "feature-importance map, model-comparison bar charts, prediction score charts, and manual-AdaBoost weight "
        "charts.\n")
    fig("feature_importance", "Pixels AdaBoost relies on most (feature importance)")

    add("## 16. Results and Discussion\n")
    if result and ev:
        conf = ctx.get("top_confusions") or []
        rep = ev["report_df"].iloc[:10]["f1-score"]
        add(f"- AdaBoost improved test accuracy from {_pct(result.baseline_test_accuracy)} (single tree) to "
            f"{_pct(result.test_accuracy)}, confirming that boosting weak learners builds a strong classifier.")
        add(f"- Training accuracy {_pct(result.train_accuracy)} vs test {_pct(result.test_accuracy)}: a gap of "
            f"{result.train_accuracy - result.test_accuracy:.1%}.")
        add(f"- Easiest digit: {rep.idxmax()} (F1 {rep.max():.3f}); hardest: {rep.idxmin()} (F1 {rep.min():.3f}).")
        if conf:
            add("- Most frequent confusions: " + "; ".join(f"{a}→{p} ({n})" for a, p, n in conf[:4])
                + ". These pairs share many pixels (e.g. 4/9 differ mainly by a closed top).")
        add("- Accuracy rises quickly over the first boosting rounds and then levels off.\n")

    add("## 17. Limitations\n")
    for s in ["Pixel-based trees ignore 2-D structure: shifted, rotated or unusually thick/thin digits are misread.",
              "AdaBoost trains sequentially and slows down with large training sets or deep trees.",
              "Sensitive to noisy or mislabelled training samples (they receive ever-larger weights).",
              "Mouse-drawn digits differ from scanned pen strokes, so live accuracy can be lower than test accuracy.",
              "The weighted vote share shown with a prediction is not a calibrated probability."]:
        add(f"- {s}")
    add("")
    add("## 18. Conclusion\n")
    add("AdaBoost with shallow decision trees is an effective and interpretable approach to handwritten digit "
        "recognition, far exceeding a single tree. The interactive application makes every stage — data, "
        "preprocessing, training, evaluation, comparison and the algorithm's mathematics — visible and reproducible.\n")
    add("## 19. Future Scope\n")
    for s in ["Feature engineering (HOG, PCA) to give trees shape-aware inputs.",
              "Probability calibration (CalibratedClassifierCV) for meaningful confidence values.",
              "Data augmentation (shifts/rotations) to improve robustness to drawing style.",
              "Gradient boosting (XGBoost/LightGBM) and a CNN baseline for comparison.",
              "Deploying the app online for public use."]:
        add(f"- {s}")
    add("")
    add("## 20. References\n")
    for s in ["Y. LeCun, C. Cortes, C. J. C. Burges — The MNIST database of handwritten digits.",
              "Y. Freund, R. E. Schapire (1997) — A decision-theoretic generalization of on-line learning and an "
              "application to boosting. Journal of Computer and System Sciences, 55(1).",
              "J. Zhu, H. Zou, S. Rosset, T. Hastie (2009) — Multi-class AdaBoost. Statistics and Its Interface, 2(3).",
              "F. Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python. JMLR 12.",
              "Streamlit and Plotly documentation."]:
        add(f"1. {s}")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# HTML and PDF
# --------------------------------------------------------------------------- #
def markdown_to_html(md_text: str, base_dir: Path | None = None, embed_images: bool = True) -> str:
    try:
        import markdown

        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        import html

        body = f"<pre>{html.escape(md_text)}</pre>"
    if embed_images and base_dir is not None:
        import base64

        def _embed(m):
            p = Path(base_dir) / m.group(1)
            if p.exists():
                return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
            return m.group(0)

        body = re.sub(r'src="([^"]+\.png)"', _embed, body)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>MNIST AdaBoost Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#24252B;line-height:1.55}}
h1{{color:#1E1F24}}h2{{color:#4B32B5;border-bottom:2px solid #C5EE5B;padding-bottom:.2rem;margin-top:2rem}}
table{{border-collapse:collapse;margin:.8rem 0;font-size:.9rem}}td,th{{border:1px solid #ddd;padding:.35rem .6rem}}
th{{background:#EEEAFE}}img{{max-width:100%}}pre{{background:#F4F4F7;padding:.8rem;border-radius:8px;overflow:auto}}
@media print{{body{{margin:0}}h2{{page-break-after:avoid}}img{{page-break-inside:avoid}}}}</style></head>
<body>{body}</body></html>"""


def markdown_to_pdf(md_text: str, pdf_path: Path, base_dir: Path) -> Path | None:
    """Render our own report Markdown to PDF with ReportLab (returns None if unavailable)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None

    # ReportLab's built-in fonts lack Greek letters and box-drawing characters (ε, α, ─),
    # so register the DejaVu fonts that ship with Matplotlib.
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular, bold, mono = "Helvetica", "Helvetica-Bold", "Courier"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(font_dir / "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuMono", str(font_dir / "DejaVuSansMono.ttf")))
        addMapping("DejaVu", 0, 0, "DejaVu")
        addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
        addMapping("DejaVu", 1, 1, "DejaVu-Bold")
        regular, bold, mono = "DejaVu", "DejaVu-Bold", "DejaVuMono"
    except Exception:
        pass  # fall back to built-in fonts (some symbols may not render)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=bold, fontSize=20, textColor=colors.HexColor(CHARCOAL))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold, textColor=colors.HexColor("#4B32B5"),
                        spaceBefore=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontName=regular, fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)

    def inline(text: str) -> str:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"`(.+?)`", rf"<font face='{mono}'>\1</font>", text)
        return text.replace("  \n", "<br/>")

    story = []
    lines = md_text.split("\n")
    i = 0
    width = A4[0] - 4 * cm
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1:j]), ParagraphStyle("pre", fontName=mono, fontSize=7.5,
                                                                                 leading=9, backColor="#F4F4F7")))
            i = j + 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append([Paragraph(inline(c), small) for c in cells])
                i += 1
            t = Table(rows, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEAFE")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 6)]
            continue
        m = re.match(r"!\[(.*?)\]\((.+?)\)", line)
        if m:
            p = Path(base_dir) / m.group(2)
            if p.exists():
                img = RLImage(str(p))
                scale = min(1.0, width / img.drawWidth, 9 * cm / img.drawHeight)
                img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
                story.append(img)
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
        elif re.match(r"^(- |\d+\. )", line):
            story.append(Paragraph("• " + inline(re.sub(r"^(- |\d+\. )", "", line)), body))
        elif line.strip() == "---":
            story.append(Spacer(1, 10))
        elif line.strip():
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||- |\d+\. |!\[|```)", lines[i + 1]):
                i += 1
                para.append(lines[i])
            story.append(Paragraph(inline("\n".join(para)), body))
        i += 1

    pdf_path = Path(pdf_path)
    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="MNIST AdaBoost Report").build(story)
    return pdf_path


def generate_report(ctx: dict, out_dir: Path) -> dict:
    """Write report.md, report.html, report.pdf (if ReportLab is available) and figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(ctx, out_dir)
    md = build_markdown(ctx, figures)
    paths = {"md": out_dir / "project_report.md", "html": out_dir / "project_report.html"}
    paths["md"].write_text(md, encoding="utf-8")
    paths["html"].write_text(markdown_to_html(md, out_dir), encoding="utf-8")
    pdf = markdown_to_pdf(md, out_dir / "project_report.pdf", out_dir)
    if pdf is not None:
        paths["pdf"] = pdf
    return paths


def build_context(data, validation, result=None, evaluation=None, comparison: pd.DataFrame | None = None,
                  experiments=None, author: str = "", institute: str = "") -> dict:
    from .evaluation import top_confusions
    from .manual_adaboost import DEFAULT_LEARNERS, DEFAULT_Y, run_manual_adaboost

    return {
        "data": data, "validation": validation, "result": result, "evaluation": evaluation,
        "comparison": comparison, "experiments": experiments or [],
        "manual": run_manual_adaboost(DEFAULT_Y, DEFAULT_LEARNERS),
        "top_confusions": top_confusions(evaluation["confusion_matrix"], 5) if evaluation else [],
        "author": author, "institute": institute,
    }
