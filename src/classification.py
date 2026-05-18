"""Classification benchmark for the Online Shoppers dataset.

Defines ten classifiers and runs grid-search + evaluation across all of them.
Plotting is delegated to ``evaluation.py`` to keep this module focused on
model training. The notebook ``01_classification.ipynb`` calls both modules.

Algorithms (10, matching the assignment rubric)
------------------------------------------------
1.  Logistic Regression                 (linear, discriminative)
2.  K-Nearest Neighbours                (instance-based)
3.  Decision Tree                       (tree)
4.  Random Forest                       (tree ensemble — bagging)
5.  Support Vector Machine (RBF kernel) (kernel)
6.  Gaussian Naive Bayes                (probabilistic, generative)
7.  Linear Discriminant Analysis        (probabilistic, generative)
8.  Gradient Boosting                   (tree ensemble — boosting)
9.  AdaBoost                            (tree ensemble — boosting)
10. Multi-Layer Perceptron              (neural network)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Algorithm catalogue with hyperparameter grids
# ---------------------------------------------------------------------------

def build_classifier_catalogue(random_state: int = 42) -> dict:
    """Return a dict mapping model name → (estimator, parameter grid).

    Hyperparameter grids are intentionally compact so the full benchmark
    finishes in 10–15 minutes on a free Colab CPU runtime.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            },
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=random_state),
            {
                "C": [1.0, 10.0],
                "gamma": ["scale"],
            },
        ),
        "Gaussian Naive Bayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            {
                "solver": ["svd"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.5, 1.0],
            },
        ),
        "MLP": (
            MLPClassifier(max_iter=500, random_state=random_state),
            {
                "hidden_layer_sizes": [(50,), (100,)],
                "alpha": [1e-4, 1e-3],
                "learning_rate_init": [1e-3],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class ClassificationBenchmarkResult:
    """Bundle returned by :func:`run_classification_benchmark`."""
    results_df: pd.DataFrame
    fitted_models: dict
    best_params: dict


def run_classification_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> ClassificationBenchmarkResult:
    """Run grid search + final evaluation for all 10 classifiers.

    For each classifier:
      1. ``GridSearchCV`` over the parameter grid using stratified k-fold CV.
      2. Best estimator is refit on the full training set.
      3. Predictions are made on the test set; metrics are computed.
      4. Training time (the whole grid search) and inference time are logged.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Preprocessed feature matrices.
    y_train, y_test : np.ndarray
        Target arrays (0/1).
    cv_folds : int, default 5
        Number of cross-validation folds.
    random_state : int, default 42
        Seed for reproducibility.
    verbose : bool, default True
        Print progress per algorithm.

    Returns
    -------
    ClassificationBenchmarkResult
        ``.results_df``   — metrics table (one row per model).
        ``.fitted_models`` — dict of best estimators (for ROC/CM plots).
        ``.best_params``  — dict of best hyperparameters per model.
    """
    catalogue = build_classifier_catalogue(random_state=random_state)
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    rows = []
    fitted_models = {}
    best_params = {}

    for name, (estimator, grid) in catalogue.items():
        if verbose:
            print(f"\n▶ {name} — grid size {_grid_size(grid)}")

        # ---- Grid search with timing ----
        t0 = time.time()
        gs = GridSearchCV(
            estimator,
            param_grid=grid,
            cv=cv,
            scoring="f1",   # F1 is appropriate for our imbalanced setting
            n_jobs=-1,      # parallelise across all available CPU cores
            refit=True,
        )
        gs.fit(X_train, y_train)
        train_time = time.time() - t0

        # ---- Inference on test set ----
        t0 = time.time()
        y_pred = gs.predict(X_test)
        inference_time = time.time() - t0

        # ---- Probability scores for ROC-AUC ----
        if hasattr(gs.best_estimator_, "predict_proba"):
            y_score = gs.predict_proba(X_test)[:, 1]
        elif hasattr(gs.best_estimator_, "decision_function"):
            y_score = gs.decision_function(X_test)
        else:
            y_score = y_pred  # fallback (shouldn't trigger for our models)

        roc_auc = roc_auc_score(y_test, y_score)

        # ---- Collect metrics ----
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc,
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "cv_best_f1": gs.best_score_,
        })

        fitted_models[name] = gs.best_estimator_
        best_params[name] = gs.best_params_

        if verbose:
            print(
                f"  ✓ acc={rows[-1]['accuracy']:.3f}  "
                f"f1={rows[-1]['f1']:.3f}  "
                f"auc={rows[-1]['roc_auc']:.3f}  "
                f"({train_time:.1f}s)"
            )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

    return ClassificationBenchmarkResult(
        results_df=results_df,
        fitted_models=fitted_models,
        best_params=best_params,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_size(grid: dict) -> int:
    """Count the number of hyperparameter combinations in a grid."""
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def best_params_to_dataframe(best_params: dict) -> pd.DataFrame:
    """Reshape the best_params dict into a tidy DataFrame for the report."""
    rows = []
    for model_name, params in best_params.items():
        for param_name, value in params.items():
            rows.append({
                "model": model_name,
                "hyperparameter": param_name,
                "best_value": str(value),
            })
    return pd.DataFrame(rows)
