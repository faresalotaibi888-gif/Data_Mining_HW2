"""Preprocessing pipelines for the CSC 588 Data Mining project.

This module is responsible for turning a raw pandas DataFrame loaded by
``data_loader.py`` into model-ready arrays that scikit-learn classifiers
can consume directly.

For the Online Shoppers Purchasing Intention dataset the pipeline performs:

1. Separating the target column ``Revenue`` from the feature matrix.
2. Identifying numeric, true-categorical, and "fake-numeric" categorical
   columns (e.g. ``OperatingSystems`` is stored as an int but represents a
   category, not a quantity).
3. Building a ``ColumnTransformer`` that scales numerics with
   ``StandardScaler`` and one-hot encodes categoricals.
4. Producing a stratified train/test split that preserves the ~15/85
   class ratio in both sides.
5. Optionally applying SMOTE to the training set only, to address the
   strong class imbalance observed during exploration.

The functions here are intentionally small and composable: each step can
be inspected or reused independently in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# SMOTE is optional. We try to import it but degrade gracefully if not
# installed so the module is still useful for users who don't want SMOTE.
try:
    from imblearn.over_sampling import SMOTE

    _HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    _HAS_IMBLEARN = False


# ---------------------------------------------------------------------------
# Column-group definitions for the Online Shoppers dataset
# ---------------------------------------------------------------------------

#: Continuous numeric columns — scaled with StandardScaler.
SHOPPERS_NUMERIC_COLS: list[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

#: Text-based categorical columns — one-hot encoded.
SHOPPERS_TEXT_CATEGORICAL_COLS: list[str] = [
    "Month",
    "VisitorType",
]

#: Columns stored as integers in the raw CSV but that actually encode
#: anonymised categories (e.g. OS #1 vs OS #2 has no numeric ordering).
SHOPPERS_FAKE_NUMERIC_CATEGORICAL_COLS: list[str] = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

#: The Weekend column is already 0/1 — no transform needed.
SHOPPERS_PASSTHROUGH_COLS: list[str] = [
    "Weekend",
]

#: Name of the target column.
SHOPPERS_TARGET_COL: str = "Revenue"


# ---------------------------------------------------------------------------
# Return-type container
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Container for the output of :func:`prepare_online_shoppers`.

    Attributes
    ----------
    X_train, X_test : np.ndarray
        Transformed feature matrices.
    y_train, y_test : np.ndarray
        Binary target vectors.
    preprocessor : ColumnTransformer
        The fitted preprocessor — kept so callers can transform new data
        consistently (e.g. user input in the Gradio UI).
    feature_names : list[str]
        Names of the columns in ``X_train``/``X_test`` after one-hot
        encoding, useful for feature importance plots.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_shoppers_preprocessor() -> ColumnTransformer:
    """Construct (but do not fit) the ColumnTransformer for Online Shoppers.

    The transformer applies, in parallel:

    * ``StandardScaler`` to continuous numeric columns.
    * ``OneHotEncoder`` to text-based categorical columns.
    * ``OneHotEncoder`` to fake-numeric categorical columns.
    * ``"passthrough"`` to the already-binary ``Weekend`` column.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessor ready for ``fit_transform``.
    """
    # handle_unknown='ignore' protects us at inference time if a category
    # we never saw during training appears in new data.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SHOPPERS_NUMERIC_COLS),
            ("cat_text", ohe, SHOPPERS_TEXT_CATEGORICAL_COLS),
            ("cat_fake_num", ohe, SHOPPERS_FAKE_NUMERIC_CATEGORICAL_COLS),
            ("pass", "passthrough", SHOPPERS_PASSTHROUGH_COLS),
        ],
        remainder="drop",   # any column not listed above is silently dropped
        verbose_feature_names_out=False,
    )


def split_features_target(
    df: pd.DataFrame,
    target_col: str = SHOPPERS_TARGET_COL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a raw dataframe into a feature matrix and a target array.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe from ``load_online_shoppers``.
    target_col : str
        The column to extract as the target.

    Returns
    -------
    X : pd.DataFrame
        The feature matrix (``df`` minus the target column).
    y : np.ndarray
        The target column as a 0/1 integer array.
    """
    X = df.drop(columns=[target_col])
    # Revenue is stored as bool — convert to {0, 1} integers for sklearn.
    y = df[target_col].astype(int).to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# End-to-end preparation
# ---------------------------------------------------------------------------

def prepare_online_shoppers(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
) -> PreparedData:
    """Run the full preprocessing pipeline for the Online Shoppers dataset.

    Steps
    -----
    1. Separate target.
    2. Stratified train/test split (preserves the ~15/85 class ratio).
    3. Fit the preprocessor on the training set, transform both sets.
    4. Optionally apply SMOTE on the training set only.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe returned by ``load_online_shoppers()``.
    test_size : float, default 0.2
        Fraction of the data held out for testing.
    random_state : int, default 42
        Seed for reproducibility.
    apply_smote : bool, default True
        Whether to apply SMOTE oversampling to the training set. If
        ``imbalanced-learn`` is not installed this is silently skipped.

    Returns
    -------
    PreparedData
        Bundle of train/test arrays plus the fitted preprocessor.
    """
    # 1) Split features from target
    X, y = split_features_target(df)

    # 2) Stratified train/test split — keeps the class ratio in both sides
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3) Build the preprocessor, fit it on training data ONLY, then
    #    transform both sets. Fitting on train-only is critical to avoid
    #    information leakage from the test set.
    preprocessor = build_shoppers_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Capture the post-encoding feature names for later (e.g. feature
    # importance plots).
    feature_names = list(preprocessor.get_feature_names_out())

    # 4) Apply SMOTE on the training set only — never on test data.
    if apply_smote and _HAS_IMBLEARN:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Convenience reporting
# ---------------------------------------------------------------------------

def describe_prepared_data(prepared: PreparedData) -> None:
    """Print a short summary of a :class:`PreparedData` object."""
    print("=== Prepared data summary ===")
    print(f"X_train shape : {prepared.X_train.shape}")
    print(f"X_test  shape : {prepared.X_test.shape}")
    print(f"y_train shape : {prepared.y_train.shape}")
    print(f"y_test  shape : {prepared.y_test.shape}")
    print(f"Features after encoding : {len(prepared.feature_names)}")

    train_pos = int(np.sum(prepared.y_train == 1))
    train_neg = int(np.sum(prepared.y_train == 0))
    test_pos = int(np.sum(prepared.y_test == 1))
    test_neg = int(np.sum(prepared.y_test == 0))

    print("\nTraining class balance:")
    print(f"  Non-buyers (0): {train_neg:>6}")
    print(f"  Buyers     (1): {train_pos:>6}")
    print("\nTest class balance:")
    print(f"  Non-buyers (0): {test_neg:>6}")
    print(f"  Buyers     (1): {test_pos:>6}")


if __name__ == "__main__":
    # Smoke test — run with: python -m src.preprocessing
    from data_loader import load_online_shoppers

    df = load_online_shoppers()
    prepared = prepare_online_shoppers(df)
    describe_prepared_data(prepared)
