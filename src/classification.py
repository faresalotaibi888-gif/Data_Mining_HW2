"""Classification benchmark for the CSC 588 Data Mining project.

Runs 10 supervised classification algorithms on a preprocessed dataset
(Online Shoppers Purchasing Intention), tunes each via ``GridSearchCV``
with stratified 5-fold cross-validation, evaluates on a held-out test
set, and saves all results (CSV + figures) under ``results/``.

Algorithms (in order)
---------------------
1.  Logistic Regression
2.  K-Nearest Neighbors
3.  Decision Tree
4.  Random Forest
5.  Support Vector Machine (RBF kernel)
6.  Gaussian Naive Bayes
7.  Linear Discriminant Analysis
8.  Gradient Boosting
9.  AdaBoost
10. Multi-Layer Perceptron

Outputs written to disk
-----------------------
* ``results/tables/classification_results.csv``
* ``results/figures/classification_metrics_comparison.png``
* ``results/figures/classification_roc_curves.png``
* ``results/figures/classification_confusion_matrices.png``
* ``results/figures/classification_training_time.png``
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FIG_METRICS_COMPARISON = FIGURES_DIR / "classification_metrics_comparison.png"
FIG_ROC_CURVES = FIGURES_DIR / "classification_roc_curves.png"
FIG_CONFUSION_MATRICES = FIGURES_DIR / "classification_confusion_matrices.png"
FIG_TRAINING_TIME = FIGURES_DIR / "classification_training_time.png"
TABLE_RESULTS = TABLES_DIR / "classification_results.csv"


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

ALGORITHMS: dict[str, tuple[Any, dict[str, Any]]] = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0]},
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(n_jobs=-1),
        {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [5, 10, None], "min_samples_split": [2, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [100], "max_depth": [10, None]},
    ),
    "SVM (RBF)": (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {"C": [1.0], "gamma": ["scale"]},
    ),
    "Gaussian NB": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "Linear Discriminant Analysis": (
        LinearDiscriminantAnalysis(),
        {"solver": ["svd"]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100], "max_depth": [3, 5]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    ),
    "MLP (Neural Net)": (
        MLPClassifier(random_state=RANDOM_STATE, max_iter=300, early_stopping=True),
        {"hidden_layer_sizes": [(50,), (100,)]},
    ),
}


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_classification_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "f1",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train, tune and evaluate all 10 classifiers.

    Returns
    -------
    results_df : pd.DataFrame
        One row per algorithm with all metrics + best hyperparameters.
        Sorted by Test F1 descending (best on top).
    fitted_models : dict[str, estimator]
        Mapping ``{algorithm_name: best_estimator}`` for downstream
        plotting and the Gradio UI.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_models: dict[str, Any] = {}

    for name, (estimator, param_grid) in ALGORITHMS.items():
        if verbose:
            print(f"[{name}] training...", flush=True)

        grid = GridSearchCV(
            estimator, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - t0

        best = grid.best_estimator_
        fitted_models[name] = best

        t1 = time.time()
        y_pred = best.predict(X_test)
        predict_time = time.time() - t1

        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test)[:, 1]
        elif hasattr(best, "decision_function"):
            y_proba = best.decision_function(X_test)
        else:
            y_proba = y_pred

        rows.append({
            "Algorithm": name,
            "Best Params": grid.best_params_,
            "CV F1 (mean)": grid.best_score_,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision": precision_score(y_test, y_pred),
            "Test Recall": recall_score(y_test, y_pred),
            "Test F1": f1_score(y_test, y_pred),
            "Test ROC-AUC": roc_auc_score(y_test, y_proba),
            "Train Time (s)": train_time,
            "Predict Time (s)": predict_time,
        })

        if verbose:
            r = rows[-1]
            print(
                f"  ✓ F1={r['Test F1']:.3f}  "
                f"AUC={r['Test ROC-AUC']:.3f}  "
                f"time={train_time:.1f}s"
            )

    results_df = pd.DataFrame(rows).sort_values(
        by="Test F1", ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(TABLE_RESULTS, index=False)
    if verbose:
        print(f"\nResults table saved → {TABLE_RESULTS}")

    return results_df, fitted_models


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics_comparison(results_df: pd.DataFrame) -> Path:
    """Grouped bar chart comparing the five test-set metrics."""
    metric_cols = [
        "Test Accuracy", "Test Precision", "Test Recall",
        "Test F1", "Test ROC-AUC",
    ]
    long = results_df.melt(
        id_vars="Algorithm", value_vars=metric_cols,
        var_name="Metric", value_name="Score",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(data=long, x="Algorithm", y="Score", hue="Metric")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Classification — Performance Comparison Across Algorithms")
    plt.ylabel("Score")
    plt.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIG_METRICS_COMPARISON, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_METRICS_COMPARISON


def plot_training_time(results_df: pd.DataFrame) -> Path:
    """Horizontal bar chart of training time per algorithm."""
    sorted_df = results_df.sort_values("Train Time (s)")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=sorted_df, y="Algorithm", x="Train Time (s)",
        hue="Algorithm", palette="rocket", legend=False,
    )
    plt.title("Classification — Training Time per Algorithm")
    plt.xlabel("Train Time (seconds, including GridSearchCV)")
    plt.tight_layout()
    plt.savefig(FIG_TRAINING_TIME, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_TRAINING_TIME


def plot_roc_curves(
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Path:
    """Overlay ROC curves for every model on a single chart."""
    plt.figure(figsize=(9, 7))

    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", linewidth=1.6)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classification — ROC Curves")
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_ROC_CURVES, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_ROC_CURVES


def plot_confusion_matrices(
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Path:
    """Grid of confusion matrices, one per algorithm."""
    n = len(fitted_models)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    axes = np.array(axes).flatten()

    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
            xticklabels=["Non-buyer", "Buyer"],
            yticklabels=["Non-buyer", "Buyer"],
        )
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Classification — Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_CONFUSION_MATRICES, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_CONFUSION_MATRICES


# ---------------------------------------------------------------------------
# Run-everything convenience
# ---------------------------------------------------------------------------

def generate_all_classification_outputs(
    results_df: pd.DataFrame,
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Path]:
    """Produce every figure used in the report and return their paths."""
    paths = {
        "metrics": plot_metrics_comparison(results_df),
        "training_time": plot_training_time(results_df),
        "roc": plot_roc_curves(fitted_models, X_test, y_test),
        "confusion": plot_confusion_matrices(fitted_models, X_test, y_test),
    }
    print("\nSaved figures:")
    for key, p in paths.items():
        print(f"  {key:14s} → {p}")
    return paths
