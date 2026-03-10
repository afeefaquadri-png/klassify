"""
Klassify – Visualization / Plot Engine.

All chart generation lives here.  Returns Plotly ``Figure`` objects
(JSON-serialisable and renderable in Streamlit with ``st.plotly_chart``).

Covers:
  * Correlation heatmap
  * Feature distributions
  * Class balance bar chart
  * ROC curves (single + multi-model)
  * Precision-Recall curves
  * Confusion matrix heatmap
  * Feature importance bar chart
  * Model comparison radar / bar
  * PCA scatter
  * Decision boundary (2-D)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA

from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_feature_distributions(df: pd.DataFrame, max_cols: int = 16) -> go.Figure:
    """Histogram grid for numeric features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    n = len(numeric_cols)
    if n == 0:
        return _empty_figure("No numeric columns found.")
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=numeric_cols)
    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, ncols)
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, showlegend=False,
                         marker_color="#636EFA"),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(title_text="Feature Distributions", height=220 * nrows)
    return fig


def plot_class_balance(class_dist: Dict[str, int]) -> go.Figure:
    """Bar chart of class frequencies."""
    labels = list(class_dist.keys())
    counts = list(class_dist.values())
    fig = px.bar(
        x=labels, y=counts,
        labels={"x": "Class", "y": "Count"},
        title="Class Balance",
        color=labels,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    """2-D PCA scatter coloured by class."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    labels = [class_names[i] if class_names else str(i) for i in y]
    fig = px.scatter(
        x=X2[:, 0], y=X2[:, 1],
        color=labels,
        labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"},
        title="PCA – 2D Projection",
        opacity=0.7,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    cm_arr = np.array(cm)
    labels = class_names if class_names else [str(i) for i in range(len(cm_arr))]
    # Normalise for colour scale
    cm_norm = cm_arr.astype(float) / (cm_arr.sum(axis=1, keepdims=True) + 1e-9)
    text = [[f"{cm_arr[i][j]}<br>({cm_norm[i][j]:.1%})"
             for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=labels, y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
    )
    return fig


def plot_roc_curves(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Overlay ROC curves for multiple models."""
    fig = go.Figure()
    for model_key, metrics in results.items():
        roc = metrics.get("roc_curve")
        auc = metrics.get("roc_auc")
        if roc is None:
            continue
        label = f"{model_key} (AUC={auc:.3f})" if auc else model_key
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines", name=label,
        ))
    # Diagonal
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode="lines", line=dict(dash="dash", color="grey"),
                             name="Random", showlegend=False))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def plot_pr_curves(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Overlay Precision-Recall curves for multiple models."""
    fig = go.Figure()
    for model_key, metrics in results.items():
        pr = metrics.get("pr_curve")
        ap = pr.get("average_precision") if pr else None
        if pr is None:
            continue
        label = f"{model_key} (AP={ap:.3f})" if ap else model_key
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"],
            mode="lines", name=label,
        ))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> go.Figure:
    """Grouped bar chart comparing multiple models across metrics."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_keys = list(results.keys())
    fig = go.Figure()
    for metric in metrics:
        values = [results[k].get(metric) or 0 for k in model_keys]
        fig.add_trace(go.Bar(name=metric, x=model_keys, y=values))
    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        yaxis=dict(range=[0, 1.05]),
        xaxis_title="Model",
        yaxis_title="Score",
    )
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, top_n * 22),
    )
    return fig


def plot_decision_boundary(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    resolution: int = 200,
) -> go.Figure:
    """
    Decision boundary plot for 2-D feature spaces.
    Projects to 2 PCA components if ``X.shape[1] > 2``.
    """
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        xlabel = "PC1"
        ylabel = "PC2"
    else:
        X2 = X
        xlabel = (feature_names[0] if feature_names else "Feature 1")
        ylabel = (feature_names[1] if feature_names else "Feature 2")

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # For PCA-projected X we need to retrain on X2 (boundary only for visual)
    from sklearn.base import clone
    vis_model = clone(model)
    vis_model.fit(X2, y)
    Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale="RdBu",
        showscale=False,
        opacity=0.4,
        contours=dict(coloring="fill"),
    ))
    unique_classes = np.unique(y)
    colors = px.colors.qualitative.Plotly
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        fig.add_trace(go.Scatter(
            x=X2[mask, 0], y=X2[mask, 1],
            mode="markers",
            name=label,
            marker=dict(color=colors[i % len(colors)], size=6, opacity=0.8),
        ))
    fig.update_layout(
        title="Decision Boundary",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_figure(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
