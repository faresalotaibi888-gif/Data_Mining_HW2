"""Association rule mining for the CSC 588 Data Mining project.

Runs two frequent-itemset algorithms — Apriori and FP-Growth — on the
Online Retail II basket data, compares their runtime, mines association
rules from the resulting itemsets, and saves comparison artefacts to
``results/``.

Outputs written to disk
-----------------------
* ``results/tables/association_frequent_itemsets.csv``
* ``results/tables/association_rules.csv``
* ``results/figures/association_top_rules_by_lift.png``
* ``results/figures/association_runtime_comparison.png``
* ``results/figures/association_support_confidence_scatter.png``
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FIG_TOP_RULES = FIGURES_DIR / "association_top_rules_by_lift.png"
FIG_RUNTIME = FIGURES_DIR / "association_runtime_comparison.png"
FIG_SUPPORT_CONFIDENCE = FIGURES_DIR / "association_support_confidence_scatter.png"
TABLE_ITEMSETS = TABLES_DIR / "association_frequent_itemsets.csv"
TABLE_RULES = TABLES_DIR / "association_rules.csv"


# ---------------------------------------------------------------------------
# Frequent itemsets — Apriori vs FP-Growth
# ---------------------------------------------------------------------------

def mine_frequent_itemsets(
    basket: pd.DataFrame,
    min_support: float = 0.01,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Mine frequent itemsets with both Apriori and FP-Growth, time each.

    Parameters
    ----------
    basket : pd.DataFrame
        One-hot basket matrix from ``preprocess_online_retail``.
    min_support : float, default 0.01
        Minimum support threshold for an itemset to be considered frequent.
        Retail basket data is sparse (thousands of distinct products), so a
        low threshold (~0.01) is needed to surface meaningful product pairs;
        higher values such as 0.03 yield very few rules.

    Returns
    -------
    apriori_itemsets : pd.DataFrame
    fpgrowth_itemsets : pd.DataFrame
    runtimes : dict[str, float]
        ``{"Apriori": seconds, "FP-Growth": seconds}``
    """
    if verbose:
        print(f"Mining frequent itemsets with min_support={min_support} ...")

    # Convert to boolean for mlxtend
    basket_bool = basket.astype(bool)

    runtimes: dict[str, float] = {}

    # Apriori
    if verbose:
        print("  [Apriori] running...", flush=True)
    t0 = time.time()
    apri_sets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    runtimes["Apriori"] = time.time() - t0
    if verbose:
        print(f"    ✓ {len(apri_sets)} itemsets in {runtimes['Apriori']:.2f}s")

    # FP-Growth
    if verbose:
        print("  [FP-Growth] running...", flush=True)
    t0 = time.time()
    fp_sets = fpgrowth(basket_bool, min_support=min_support, use_colnames=True)
    runtimes["FP-Growth"] = time.time() - t0
    if verbose:
        print(f"    ✓ {len(fp_sets)} itemsets in {runtimes['FP-Growth']:.2f}s")

    speedup = runtimes["Apriori"] / max(runtimes["FP-Growth"], 1e-9)
    if verbose:
        print(f"\n  FP-Growth speedup vs Apriori: {speedup:.1f}x")

    # Persist (use the larger set — both should be identical content)
    larger = fp_sets if len(fp_sets) >= len(apri_sets) else apri_sets
    larger.sort_values("support", ascending=False).to_csv(TABLE_ITEMSETS, index=False)
    if verbose:
        print(f"  Itemsets table saved → {TABLE_ITEMSETS}")

    return apri_sets, fp_sets, runtimes


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------

def mine_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.3,
    top_n: int | None = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Derive association rules from frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : pd.DataFrame
        Output of ``apriori`` or ``fpgrowth`` (they should agree).
    min_confidence : float, default 0.3
        Lower bound on rule confidence.
    top_n : int or None
        If given, keep only the top-N rules by lift in the saved CSV.

    Returns
    -------
    pd.DataFrame
        Rules sorted by lift, descending.
    """
    if verbose:
        print(f"Generating rules with min_confidence={min_confidence} ...")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"  ✓ {len(rules)} rules generated")

    # Convert frozenset to readable strings for the CSV / report
    rules_export = rules.copy()
    rules_export["antecedents"] = rules_export["antecedents"].apply(
        lambda s: ", ".join(sorted(s))
    )
    rules_export["consequents"] = rules_export["consequents"].apply(
        lambda s: ", ".join(sorted(s))
    )

    if top_n is not None:
        rules_export = rules_export.head(top_n)

    rules_export.to_csv(TABLE_RULES, index=False)
    if verbose:
        print(f"  Rules table saved → {TABLE_RULES}")

    return rules


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_runtime_comparison(runtimes: dict[str, float]) -> Path:
    """Bar chart comparing Apriori vs FP-Growth runtimes."""
    plt.figure(figsize=(8, 5))
    algos = list(runtimes.keys())
    times = list(runtimes.values())
    bars = plt.bar(
        algos, times,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black", linewidth=1.2,
    )

    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{t:.2f}s",
            ha="center", fontsize=11, fontweight="bold",
        )

    speedup = runtimes["Apriori"] / max(runtimes["FP-Growth"], 1e-9)
    plt.title(
        f"Frequent-Itemset Mining — Apriori vs FP-Growth\n"
        f"(FP-Growth speedup: {speedup:.1f}×)"
    )
    plt.ylabel("Runtime (seconds)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_RUNTIME, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_RUNTIME


def plot_top_rules_by_lift(rules: pd.DataFrame, top_n: int = 15) -> Path:
    """Horizontal bar chart of the top-N rules ranked by lift."""
    top = rules.head(top_n).copy()
    top["rule"] = top.apply(
        lambda r: f"{', '.join(sorted(r['antecedents']))}  →  "
                  f"{', '.join(sorted(r['consequents']))}",
        axis=1,
    )

    plt.figure(figsize=(12, max(5, 0.45 * top_n)))
    sns.barplot(
        data=top, x="lift", y="rule",
        hue="rule", palette="viridis", legend=False,
    )
    plt.title(f"Top {top_n} Association Rules by Lift")
    plt.xlabel("Lift")
    plt.ylabel("")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_TOP_RULES, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_TOP_RULES


def plot_support_confidence_scatter(rules: pd.DataFrame) -> Path:
    """Scatter plot: support vs. confidence, colour = lift, size = lift."""
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="viridis",
        s=20 + 5 * np.clip(rules["lift"], 0, 30),
        alpha=0.75, edgecolors="black", linewidths=0.4,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules — Support vs Confidence (colour = Lift)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_SUPPORT_CONFIDENCE, dpi=150, bbox_inches="tight")
    plt.show()
    return FIG_SUPPORT_CONFIDENCE


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def generate_all_association_outputs(
    rules: pd.DataFrame, runtimes: dict[str, float], top_n: int = 15
) -> dict[str, Path]:
    """Generate all figures and return their paths."""
    paths = {
        "runtime": plot_runtime_comparison(runtimes),
        "top_rules": plot_top_rules_by_lift(rules, top_n=top_n),
        "scatter": plot_support_confidence_scatter(rules),
    }
    print("\nSaved figures:")
    for k, v in paths.items():
        print(f"  {k:10s} → {v}")
    return paths
