"""Plotly figure builders. Every function takes real data and returns a figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Design tokens (shared with assets/styles.css)
CHARCOAL = "#1E1F24"
INK = "#24252B"
MUTED = "#6B6F7B"
LIME = "#C5EE5B"
LIME_DARK = "#7FA51A"
LAVENDER = "#C9BEFA"
PURPLE = "#7C5CE6"
PURPLE_DEEP = "#4B32B5"
CARD = "#FFFFFF"
GRID = "#ECECF1"
RED = "#E45B6B"

SEQ_PURPLE = [[0, "#F7F5FF"], [0.15, "#E4DCFE"], [0.4, "#B9A8F7"], [0.7, "#8467E8"], [1, "#3E2A99"]]
DIGIT_SCALE = [[0, "#111216"], [1, "#FFFFFF"]]

FONT = "Plus Jakarta Sans, Inter, Segoe UI, sans-serif"


def _style(fig: go.Figure, height: int = 320, title: str | None = None, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01) if title else None,
        showlegend=showlegend,
        hoverlabel=dict(bgcolor=CHARCOAL, font_color="white", font_family=FONT, bordercolor=CHARCOAL),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def class_distribution(train_counts, test_counts=None, height: int = 300) -> go.Figure:
    digits = [str(d) for d in range(10)]
    fig = go.Figure()
    fig.add_bar(x=digits, y=train_counts, name="Train", marker_color=PURPLE, marker_line_width=0,
                hovertemplate="Digit %{x}<br>Training images: %{y:,}<extra></extra>")
    if test_counts is not None:
        fig.add_bar(x=digits, y=test_counts, name="Test", marker_color=LIME, marker_line_width=0,
                    hovertemplate="Digit %{x}<br>Test images: %{y:,}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(title="Digit")
    fig.update_yaxes(title="Images")
    return _style(fig, height, showlegend=test_counts is not None)


def digit_grid(images, titles, cols: int = 6, height: int | None = None, title_colors=None) -> go.Figure:
    """A grid of 28x28 digits with per-image titles (e.g. labels / predictions)."""
    n = len(images)
    if n == 0:
        return _style(go.Figure(), 100)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.08)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        fig.add_trace(go.Heatmap(z=np.asarray(img)[::-1], colorscale=DIGIT_SCALE, showscale=False,
                                 hovertemplate="row %{y}, col %{x}<br>value %{z:.2f}<extra></extra>"),
                      row=r + 1, col=c + 1)
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", constrain="domain")
    for idx, ann in enumerate(fig.layout.annotations):
        ann.font = dict(size=12, color=(title_colors[idx] if title_colors else INK), family=FONT)
    fig = _style(fig, height or max(140, 125 * rows))
    fig.update_layout(margin=dict(l=4, r=4, t=26, b=4))
    return fig


def single_digit(image, title: str | None = None, height: int = 260, show_values: bool = False) -> go.Figure:
    img = np.asarray(image, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=img[::-1], colorscale=DIGIT_SCALE, showscale=False,
        text=np.round(img[::-1], 2) if show_values else None,
        texttemplate="%{text}" if show_values else None, textfont=dict(size=7),
        hovertemplate="row %{customdata}, col %{x}<br>value %{z:.3f}<extra></extra>",
        customdata=np.repeat(np.arange(img.shape[0])[::-1, None], img.shape[1], axis=1),
    ))
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", constrain="domain")
    return _style(fig, height, title)


def pixel_heatmap(image, title: str | None = None, height: int = 420, colorscale=None, zmax=None) -> go.Figure:
    """A 28x28 matrix with axis ticks and a colour bar (for inspecting pixel values)."""
    img = np.asarray(image, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=img, colorscale=colorscale or SEQ_PURPLE, zmin=0, zmax=zmax,
        colorbar=dict(thickness=10, outlinewidth=0),
        hovertemplate="row %{y}, col %{x}<br>value %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain", title="row")
    fig.update_xaxes(constrain="domain", title="column")
    return _style(fig, height, title)


def class_scores_bar(scores, predicted: int, label: str = "Score", height: int = 300) -> go.Figure:
    scores = np.asarray(scores, dtype=float)
    colors = [LIME if d == predicted else LAVENDER for d in range(10)]
    fig = go.Figure(go.Bar(
        x=[str(d) for d in range(10)], y=scores, marker_color=colors, marker_line_width=0,
        text=[f"{s:.3f}" for s in scores], textposition="outside", cliponaxis=False,
        hovertemplate="Digit %{x}<br>" + label + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Digit")
    fig.update_yaxes(title=label)
    return _style(fig, height)


def confusion_matrix_fig(cm, height: int = 520, normalize: bool = False) -> go.Figure:
    cm = np.asarray(cm)
    z = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    digits = [str(d) for d in range(10)]
    text = [[f"{v:.1%}" if normalize else f"{int(v)}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=digits, y=digits, colorscale=SEQ_PURPLE, text=text, texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(thickness=10, outlinewidth=0),
        hoverinfo="skip", xgap=2, ygap=2,
    ))
    # Heatmap cells cannot be "selected" in Plotly, so an invisible marker sits on every cell.
    # It carries the hover text and makes cells clickable (Streamlit on_select).
    gx, gy = np.meshgrid(np.arange(10), np.arange(10))
    fig.add_trace(go.Scatter(
        x=[str(v) for v in gx.ravel()], y=[str(v) for v in gy.ravel()], mode="markers",
        marker=dict(size=26, color="rgba(0,0,0,0)", symbol="square"),
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        customdata=cm.ravel(), showlegend=False,
        hovertemplate="Actual %{y} → Predicted %{x}<br>%{customdata:,} images<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted Digit", side="bottom", type="category", categoryorder="array",
                     categoryarray=digits)
    fig.update_yaxes(title="Actual Digit", autorange="reversed", type="category", categoryorder="array",
                     categoryarray=digits)
    fig.update_layout(clickmode="event+select", dragmode=False)
    return _style(fig, height)


def staged_accuracy(test_acc, train_acc=None, height: int = 320) -> go.Figure:
    rounds = np.arange(1, len(test_acc) + 1)
    fig = go.Figure()
    if train_acc:
        fig.add_scatter(x=rounds, y=train_acc, name="Training subset", mode="lines",
                        line=dict(color=LIME_DARK, width=2, dash="dot"),
                        hovertemplate="Round %{x}<br>Train accuracy %{y:.2%}<extra></extra>")
    fig.add_scatter(x=rounds, y=test_acc, name="Test set", mode="lines", line=dict(color=PURPLE, width=3),
                    fill="tozeroy", fillcolor="rgba(124,92,230,0.10)",
                    hovertemplate="Round %{x}<br>Test accuracy %{y:.2%}<extra></extra>")
    lo = min(min(test_acc), min(train_acc) if train_acc else 1)
    fig.update_yaxes(tickformat=".0%", title="Accuracy", range=[max(0, lo - 0.05), 1.0])
    fig.update_xaxes(title="Boosting round (number of weak learners)")
    return _style(fig, height, showlegend=True)


def estimator_weights_fig(weights, errors, height: int = 300) -> go.Figure:
    rounds = np.arange(1, len(weights) + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=rounds, y=weights, name="Learner weight (α)", marker_color=LAVENDER,
                hovertemplate="Round %{x}<br>α = %{y:.3f}<extra></extra>", secondary_y=False)
    fig.add_scatter(x=rounds, y=errors, name="Weighted error (ε)", mode="lines+markers",
                    line=dict(color=PURPLE_DEEP, width=2), marker=dict(size=4),
                    hovertemplate="Round %{x}<br>ε = %{y:.3f}<extra></extra>", secondary_y=True)
    fig.update_yaxes(title_text="α", secondary_y=False)
    fig.update_yaxes(title_text="ε", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Boosting round")
    return _style(fig, height, showlegend=True)


def comparison_bars(df: pd.DataFrame, metric: str, height: int = 280, fmt: str = ".2%") -> go.Figure:
    lower_is_better = "time" in metric.lower()
    best = df[metric].idxmin() if lower_is_better else df[metric].idxmax()
    colors = [LIME if i == best else (PURPLE if m == "AdaBoost" else LAVENDER) for i, m in zip(df.index, df["Model"])]
    text = [format(v, fmt) if "%" in fmt else f"{v:.2f}s" for v in df[metric]]
    fig = go.Figure(go.Bar(x=df["Model"], y=df[metric], marker_color=colors, text=text, textposition="outside",
                           cliponaxis=False, marker_line_width=0,
                           hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>"))
    if "%" in fmt:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    fig.update_yaxes(title=metric)
    return _style(fig, height, f"{metric} ({'lower' if lower_is_better else 'higher'} is better)")


def experiments_fig(rows: list[dict], height: int = 300) -> go.Figure:
    df = pd.DataFrame(rows).reset_index(drop=True)
    labels = [f"#{i + 1}: {r.n_estimators} trees, d{r.max_depth}, lr {r.learning_rate}, n={r.train_size:,}"
              for i, r in df.iterrows()]
    fig = go.Figure()
    fig.add_bar(x=[f"#{i + 1}" for i in range(len(df))], y=df["test_accuracy"], marker_color=PURPLE,
                customdata=np.stack([labels, df["train_time_s"]], axis=1), name="Test accuracy",
                hovertemplate="%{customdata[0]}<br>Test accuracy %{y:.2%}<br>Train time %{customdata[1]}s<extra></extra>")
    fig.add_scatter(x=[f"#{i + 1}" for i in range(len(df))], y=df["baseline_test_accuracy"], mode="markers",
                    marker=dict(color=LIME_DARK, size=11, symbol="diamond"), name="Single-tree baseline",
                    hovertemplate="Baseline %{y:.2%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="Test accuracy")
    fig.update_xaxes(title="Experiment")
    return _style(fig, height, showlegend=True)


def manual_weights_fig(rounds, height: int = 320) -> go.Figure:
    """Sample weights before round 1 and after each round of the manual example."""
    if not rounds:
        return _style(go.Figure(), 100)
    n = len(rounds[0].weights_before)
    samples = [f"x{i + 1}" for i in range(n)]
    palette = ["#DCD3FD", LAVENDER, PURPLE, PURPLE_DEEP, CHARCOAL]
    fig = go.Figure()
    fig.add_bar(x=samples, y=rounds[0].weights_before, name="Initial (1/N)", marker_color=LIME)
    for i, r in enumerate(rounds):
        fig.add_bar(x=samples, y=r.weights_after, name=f"After round {i + 1}", marker_color=palette[min(i + 1, 4)],
                    hovertemplate="%{x}: %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Sample weight")
    return _style(fig, height, showlegend=True)


def performance_fig(train_time: float, infer_ms: float, height: int = 240) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training time (s)", "Inference per image (ms)"))
    fig.add_bar(x=["AdaBoost"], y=[train_time], marker_color=PURPLE, row=1, col=1,
                text=[f"{train_time:.1f}s"], textposition="outside", cliponaxis=False,
                hovertemplate="%{y:.2f} s<extra></extra>")
    fig.add_bar(x=["AdaBoost"], y=[infer_ms], marker_color=LIME, row=1, col=2,
                text=[f"{infer_ms:.3f}ms"], textposition="outside", cliponaxis=False,
                hovertemplate="%{y:.4f} ms<extra></extra>")
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, height)
