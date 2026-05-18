"""Shared evaluation helpers — plots and tables — for all three project tracks.

Every plotting function in this module:
* Saves to ``results/figures/<filename>.png`` at 200 DPI.
* Returns the matplotlib Figure so notebooks can also display inline.
* Uses a consistent style (tight layout, no top/right spines).

Every table-saving function writes to ``results/tables/<filename>.csv``.

Filenames must follow the project's naming convention so report screenshots
match repository files exactly. See README.md for the convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to ``results/figures/`` and return the path."""
    if not filename.endswith(".png"):
        filename += ".png"
    path = FIGURES_DIR / filename
    fig.savefig(path)
    print(f"  📊 Saved: {path.relative_to(RESULTS_DIR.parent)}")
    return path


def _save_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a dataframe to ``results/tables/`` and return the path."""
    if not filename.endswith(".csv"):
        filename += ".csv"
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"  📋 Saved: {path.relative_to(RESULTS_DIR.parent)}")
    return path


# ---------------------------------------------------------------------------
# Comparison bar charts
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str,
    filename: str,
    title: str | None = None,
    sort: bool = True,
) -> plt.Figure:
    """Horizontal bar chart comparing one metric across algorithms.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain a ``model`` column and the requested ``metric`` column.
    metric : str
        Column name to plot (e.g. ``"f1"``, ``"accuracy"``).
    filename : str
        Output filename (without extension) — should follow naming convention.
    title : str, optional
        Chart title. Defaults to ``"<metric.title()> comparison"``.
    sort : bool, default True
        Sort bars in descending order of the metric value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = results_df.sort_values(metric, ascending=True) if sort else results_df

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["model"], df[metric], color="#3498db")

    # annotate values at the bar ends
    for bar, value in zip(bars, df[metric]):
        ax.text(value, bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}", va="center", fontsize=9)

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} comparison")
    ax.set_xlim(0, max(df[metric]) * 1.15)

    _save_fig(fig, filename)
    return fig


def plot_time_vs_accuracy(
    results_df: pd.DataFrame,
    filename: str,
    accuracy_col: str = "accuracy",
    time_col: str = "train_time_s",
) -> plt.Figure:
    """Scatter plot of training time vs accuracy — shows efficiency trade-off."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in results_df.iterrows():
        ax.scatter(row[time_col], row[accuracy_col], s=120, alpha=0.7)
        ax.annotate(row["model"],
                    (row[time_col], row[accuracy_col]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.set_title("Training time vs. accuracy trade-off")
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Classification-specific plots
# ---------------------------------------------------------------------------

def plot_roc_curves(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    filename: str,
) -> plt.Figure:
    """Overlay ROC curves for every fitted classifier on one chart."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in fitted_models.items():
        # Get prediction scores: prefer predict_proba, fall back to decision_function
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — all classifiers")
    ax.legend(loc="lower right", fontsize=9)
    _save_fig(fig, filename)
    return fig


def plot_confusion_matrices_grid(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    filename: str,
    ncols: int = 5,
) -> plt.Figure:
    """Grid of confusion matrices, one per classifier."""
    n_models = len(fitted_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (name, model) in zip(axes, fitted_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=ax,
                    xticklabels=["No buy", "Buy"],
                    yticklabels=["No buy", "Buy"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # hide any unused subplots
    for ax in axes[n_models:]:
        ax.axis("off")

    fig.suptitle("Confusion matrices on test set", y=1.02, fontsize=13)
    _save_fig(fig, filename)
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    filename: str,
    top_n: int = 20,
    title: str | None = None,
) -> plt.Figure:
    """Top-N feature importances for a tree-based model."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values, color="#27ae60")
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Top {top_n} feature importances")
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Clustering-specific plots
# ---------------------------------------------------------------------------

def plot_elbow(
    k_values: Iterable[int],
    inertias: Iterable[float],
    filename: str,
) -> plt.Figure:
    """Elbow plot of K-Means inertia vs k."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_values), list(inertias), "o-", lw=2, color="#e67e22")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow method — K-Means")
    _save_fig(fig, filename)
    return fig


def plot_silhouette_scores(
    k_values: Iterable[int],
    scores: Iterable[float],
    filename: str,
    method_name: str = "K-Means",
) -> plt.Figure:
    """Silhouette score vs k."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_values), list(scores), "s-", lw=2, color="#9b59b6")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette scores — {method_name}")
    _save_fig(fig, filename)
    return fig


def plot_dendrogram(
    X: np.ndarray,
    filename: str,
    method: str = "ward",
    max_display: int = 30,
) -> plt.Figure:
    """Plot a hierarchical clustering dendrogram (truncated)."""
    # Local import to keep scipy optional at the module level
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=max_display,
        show_leaf_counts=True,
        leaf_rotation=45,
        ax=ax,
    )
    ax.set_xlabel("Cluster size at this leaf")
    ax.set_ylabel("Linkage distance")
    ax.set_title(f"Hierarchical clustering dendrogram ({method} linkage)")
    _save_fig(fig, filename)
    return fig


def plot_dbscan_kdistance(
    distances: np.ndarray,
    suggested_eps: float,
    filename: str,
    k: int = 5,
) -> plt.Figure:
    """k-distance plot for choosing DBSCAN's eps parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, color="#34495e", lw=1.5)
    ax.axhline(
        suggested_eps, color="red", linestyle="--",
        label=f"Suggested eps = {suggested_eps:.3f}",
    )
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour")
    ax.set_title("DBSCAN — k-distance plot")
    ax.legend()
    _save_fig(fig, filename)
    return fig


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    filename: str,
    title: str = "Cluster visualization (PCA-projected)",
) -> plt.Figure:
    """Scatter plot of points in 2D coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))

    for label, color in zip(unique_labels, palette):
        mask = labels == label
        label_str = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=20, alpha=0.6, c=[color], label=label_str)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Association-specific plots
# ---------------------------------------------------------------------------

def plot_top_rules_by_lift(
    rules_df: pd.DataFrame,
    filename: str,
    top_n: int = 15,
) -> plt.Figure:
    """Horizontal bar chart of the top-N association rules by lift."""
    df = rules_df.nlargest(top_n, "lift").copy()
    df["rule"] = (
        df["antecedents"].astype(str) + " → " + df["consequents"].astype(str)
    )
    df = df.sort_values("lift", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(df["rule"], df["lift"], color="#c0392b")
    ax.set_xlabel("Lift")
    ax.set_title(f"Top {top_n} association rules by lift")
    _save_fig(fig, filename)
    return fig


def plot_lift_vs_confidence(
    rules_df: pd.DataFrame,
    filename: str,
) -> plt.Figure:
    """Scatter of association rules in (confidence, lift) space, sized by support."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        rules_df["confidence"], rules_df["lift"],
        s=rules_df["support"] * 5000,
        alpha=0.5, c=rules_df["lift"], cmap="viridis",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Lift")
    ax.set_title("Association rules — confidence × lift (size = support)")
    plt.colorbar(sc, ax=ax, label="Lift")
    _save_fig(fig, filename)
    return fig


def plot_runtime_comparison(
    runtimes: dict,
    filename: str,
    title: str = "Apriori vs FP-Growth runtime",
) -> plt.Figure:
    """Bar chart comparing wall-clock runtime of association algorithms."""
    names = list(runtimes.keys())
    times = list(runtimes.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, times, color=["#16a085", "#2980b9"])
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.2f}s",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Public table savers
# ---------------------------------------------------------------------------

def save_results_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a results table to ``results/tables/``."""
    return _save_table(df, filename)
