"""Clustering routines for the CSC 588 Data Mining project.

Three algorithms covered (matching the assignment requirement of 2-3
techniques for non-classification tracks):

* **K-Means**                   — centroid-based, hard partitioning.
* **Agglomerative Hierarchical**  — bottom-up linkage, produces a dendrogram.
* **DBSCAN**                    — density-based, discovers arbitrary shapes
                                 and labels outliers as noise (-1).

Plotting is delegated to ``evaluation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# K-Means diagnostics (for picking k)
# ---------------------------------------------------------------------------

@dataclass
class KMeansDiagnostics:
    """Sweep results for choosing k."""
    k_values: list[int]
    inertias: list[float]
    silhouettes: list[float]


def kmeans_diagnostics(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> KMeansDiagnostics:
    """Run K-Means for k in [k_min, k_max] and return inertia + silhouette.

    Use the result with ``plot_elbow`` and ``plot_silhouette_scores`` from
    ``evaluation.py`` to pick the best k.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))

    return KMeansDiagnostics(
        k_values=k_values,
        inertias=inertias,
        silhouettes=silhouettes,
    )


# ---------------------------------------------------------------------------
# DBSCAN epsilon heuristic
# ---------------------------------------------------------------------------

def suggest_dbscan_eps(
    X: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, float]:
    """Compute the sorted k-distance curve and suggest a starting eps.

    The recommended eps is near the "elbow" of the curve. We auto-pick
    the 90th-percentile point, which works well for normalised data.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k : int, default 5
        Number of neighbours (== min_samples to be used in DBSCAN).

    Returns
    -------
    distances : np.ndarray
        Sorted ascending k-distances for every point.
    suggested_eps : float
        A reasonable starting eps.
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)

    kth_dist = np.sort(dist[:, -1])
    suggested_eps = float(np.percentile(kth_dist, 90))
    return kth_dist, suggested_eps


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """Fit K-Means and return (labels, fitted_model)."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """Fit Agglomerative Hierarchical clustering and return (labels, model)."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return labels, agg


def run_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int = 5,
) -> Tuple[np.ndarray, DBSCAN]:
    """Fit DBSCAN and return (labels, model). Noise points get label -1."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db


# ---------------------------------------------------------------------------
# Internal quality metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str = "Algorithm",
) -> dict:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz.

    Noise points (label = -1, only produced by DBSCAN) are excluded
    before scoring so we don't penalise a clean clustering for having
    flagged outliers.
    """
    mask = labels != -1
    n_noise = int((~mask).sum())
    unique = np.unique(labels[mask])
    n_clusters = len(unique)

    out = {
        "algorithm": algorithm_name,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }

    if n_clusters < 2 or mask.sum() < 2:
        out.update({
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        })
        return out

    Xe = X[mask]
    le = labels[mask]
    out["silhouette"] = float(silhouette_score(Xe, le))
    out["davies_bouldin"] = float(davies_bouldin_score(Xe, le))
    out["calinski_harabasz"] = float(calinski_harabasz_score(Xe, le))
    return out


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Stack a list of metric dicts into one comparison DataFrame."""
    return pd.DataFrame(metrics_list)


# ---------------------------------------------------------------------------
# Cluster interpretation
# ---------------------------------------------------------------------------

def profile_clusters(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute the mean feature value per cluster, plus point counts.

    Used for the report: explains *what each cluster represents*
    (e.g. "Cluster 2 = high purchase frequency, low credit limit").

    Parameters
    ----------
    df_original : pd.DataFrame
        The *un-scaled* feature DataFrame so means are in original units.
    labels : np.ndarray
        Cluster assignments aligned with ``df_original``.
    feature_names : list[str], optional
        Restrict the profile to these columns. Defaults to all numeric.

    Returns
    -------
    pd.DataFrame
        Rows = clusters, columns = features, values = mean.
        Includes an ``n_points`` column with cluster size.
    """
    df = df_original.copy()
    if feature_names is not None:
        df = df[feature_names]

    df = df.select_dtypes(include=[np.number]).copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").mean()
    profile["n_points"] = df.groupby("cluster").size()
    return profile.reset_index()
