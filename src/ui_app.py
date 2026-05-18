"""Gradio user interface for the CSC 588 Data Mining project.

The UI exposes four tabs:

1. **About** — project summary and dataset descriptions.
2. **Classification** — interactive predictor for the Online Shoppers
   dataset. The user enters session attributes and selects a trained
   classifier; the app returns the probability of a purchase.
3. **Clustering** — segment a (synthetic or uploaded) customer profile
   using one of the three trained clusterers, and show its position on
   the PCA scatter.
4. **Association** — query top association rules containing a chosen
   product (autocomplete from the catalog).

The app is designed to launch inside Colab (``share=True`` gives a
public URL for the demo) but also works locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Globals — populated at launch time from the pipelines
# ---------------------------------------------------------------------------

CLASSIFICATION_STATE: dict[str, Any] = {
    "models": None,            # dict[str, estimator]
    "preprocessor": None,      # fitted ColumnTransformer
    "feature_template": None,  # one row of df.drop(['Revenue']) as a template
}

CLUSTERING_STATE: dict[str, Any] = {
    "models": None,            # dict[str, np.ndarray of labels]
    "X_scaled": None,
    "feature_names": None,
    "scaler": None,
}

ASSOCIATION_STATE: dict[str, Any] = {
    "rules": None,             # pd.DataFrame of rules
}


# ---------------------------------------------------------------------------
# Tab 1: About
# ---------------------------------------------------------------------------

ABOUT_MARKDOWN = """
# 🧠 CSC 588 — Data Mining Project

A unified benchmark covering the three major paradigms of data mining:

| Paradigm | Dataset | Algorithms |
|---|---|---|
| **Classification** | Online Shoppers Purchasing Intention (UCI) | 10 algorithms incl. RF, GB, SVM, MLP, LDA |
| **Clustering** | CC General — Credit Card Customers (Kaggle) | K-Means, Agglomerative, DBSCAN |
| **Association** | Online Retail II (UCI) | Apriori, FP-Growth |

## Hypotheses tested

- **H1** Ensembles outperform single-model classifiers on accuracy and F1.
- **H2** Tree-based ensembles offer the best accuracy/runtime trade-off on tabular data.
- **H3** Proper imbalance handling (SMOTE) improves minority-class recall.
- **H4** DBSCAN reveals non-spherical structure missed by K-Means.
- **H5** FP-Growth is substantially faster than Apriori with equivalent rules.
- **H6** Hyperparameter tuning yields measurable gains over defaults.

Use the tabs above to interact with each trained pipeline.
"""


# ---------------------------------------------------------------------------
# Tab 2: Classification — interactive predictor
# ---------------------------------------------------------------------------

def predict_session(
    administrative: float,
    administrative_duration: float,
    informational: float,
    informational_duration: float,
    product_related: float,
    product_related_duration: float,
    bounce_rates: float,
    exit_rates: float,
    page_values: float,
    special_day: float,
    month: str,
    operating_systems: int,
    browser: int,
    region: int,
    traffic_type: int,
    visitor_type: str,
    weekend: bool,
    model_name: str,
) -> str:
    """Predict whether a session ends in a purchase."""
    state = CLASSIFICATION_STATE
    if state["models"] is None:
        return "⚠️ Classification pipeline not initialised — please run the benchmark first."

    # Build a single-row DataFrame in the same shape as training
    row = state["feature_template"].copy()
    row.loc[0, "Administrative"] = administrative
    row.loc[0, "Administrative_Duration"] = administrative_duration
    row.loc[0, "Informational"] = informational
    row.loc[0, "Informational_Duration"] = informational_duration
    row.loc[0, "ProductRelated"] = product_related
    row.loc[0, "ProductRelated_Duration"] = product_related_duration
    row.loc[0, "BounceRates"] = bounce_rates
    row.loc[0, "ExitRates"] = exit_rates
    row.loc[0, "PageValues"] = page_values
    row.loc[0, "SpecialDay"] = special_day
    row.loc[0, "Month"] = month
    row.loc[0, "OperatingSystems"] = operating_systems
    row.loc[0, "Browser"] = browser
    row.loc[0, "Region"] = region
    row.loc[0, "TrafficType"] = traffic_type
    row.loc[0, "VisitorType"] = visitor_type
    row.loc[0, "Weekend"] = weekend

    # Transform with the fitted preprocessor and predict
    X = state["preprocessor"].transform(row)
    model = state["models"][model_name]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        # Map decision-function output to a pseudo-probability via sigmoid
        z = float(model.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-z))
    else:
        proba = float(model.predict(X)[0])

    label = "🟢 LIKELY TO BUY" if proba >= 0.5 else "🔴 UNLIKELY TO BUY"
    return f"**{label}**\n\nProbability of purchase: **{proba:.1%}**\n\nModel used: `{model_name}`"


def build_classification_tab() -> gr.Blocks:
    """Construct the classification tab as a Gradio Blocks layout."""
    months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    visitor_types = ["Returning_Visitor", "New_Visitor", "Other"]

    with gr.Blocks() as tab:
        gr.Markdown("### Predict Session Outcome\n"
                    "Enter session attributes and pick a model. The probability "
                    "of the session ending in a purchase will be returned.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Page interaction")
                administrative = gr.Number(value=2, label="Administrative pages")
                administrative_duration = gr.Number(value=80, label="Administrative duration (s)")
                informational = gr.Number(value=0, label="Informational pages")
                informational_duration = gr.Number(value=0, label="Informational duration (s)")
                product_related = gr.Number(value=20, label="Product-related pages")
                product_related_duration = gr.Number(value=600, label="Product-related duration (s)")

            with gr.Column():
                gr.Markdown("#### Engagement & timing")
                bounce_rates = gr.Slider(0, 0.2, value=0.02, label="Bounce rate")
                exit_rates = gr.Slider(0, 0.2, value=0.04, label="Exit rate")
                page_values = gr.Number(value=10.0, label="Page values")
                special_day = gr.Slider(0, 1, value=0.0, step=0.1, label="Special-day closeness")
                month = gr.Dropdown(months, value="Nov", label="Month")
                weekend = gr.Checkbox(value=False, label="Weekend")

            with gr.Column():
                gr.Markdown("#### Technical & visitor")
                operating_systems = gr.Number(value=2, precision=0, label="OS code")
                browser = gr.Number(value=2, precision=0, label="Browser code")
                region = gr.Number(value=1, precision=0, label="Region code")
                traffic_type = gr.Number(value=2, precision=0, label="Traffic-type code")
                visitor_type = gr.Dropdown(visitor_types, value="Returning_Visitor", label="Visitor type")
                model_name = gr.Dropdown(
                    choices=list(CLASSIFICATION_STATE["models"].keys())
                    if CLASSIFICATION_STATE["models"] else [],
                    value=None,
                    label="Model to use",
                )

        predict_btn = gr.Button("🔮 Predict", variant="primary")
        output = gr.Markdown()

        predict_btn.click(
            predict_session,
            inputs=[
                administrative, administrative_duration,
                informational, informational_duration,
                product_related, product_related_duration,
                bounce_rates, exit_rates, page_values, special_day,
                month, operating_systems, browser, region,
                traffic_type, visitor_type, weekend, model_name,
            ],
            outputs=output,
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 3: Clustering — explore segments
# ---------------------------------------------------------------------------

def get_cluster_summary(algorithm: str) -> str:
    """Return a markdown table of cluster sizes for the chosen algorithm."""
    state = CLUSTERING_STATE
    if state["models"] is None:
        return "⚠️ Clustering pipeline not initialised."

    labels = state["models"].get(algorithm)
    if labels is None:
        return f"Algorithm '{algorithm}' not found."

    unique, counts = np.unique(labels, return_counts=True)
    rows = ["| Cluster | Size | Share |", "|---|---|---|"]
    total = len(labels)
    for u, c in zip(unique, counts):
        name = "Noise" if u == -1 else f"Cluster {u}"
        rows.append(f"| {name} | {c:,} | {c/total:.1%} |")
    return "\n".join(rows)


def build_clustering_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "### Cluster Explorer\n"
            "Choose a clustering algorithm to inspect the resulting customer "
            "segments. Cluster profiles and the 2D PCA visualisation are "
            "available in the saved figures under `results/figures/`."
        )

        choices = list(CLUSTERING_STATE["models"].keys()) if CLUSTERING_STATE["models"] else []
        algo = gr.Dropdown(choices=choices, value=choices[0] if choices else None,
                           label="Clustering algorithm")
        summary = gr.Markdown()
        refresh_btn = gr.Button("🔄 Show segments")
        refresh_btn.click(get_cluster_summary, inputs=algo, outputs=summary)

    return tab


# ---------------------------------------------------------------------------
# Tab 4: Association — rule lookup
# ---------------------------------------------------------------------------

def search_rules(product_query: str, top_n: int = 10) -> pd.DataFrame:
    """Return rules whose antecedents or consequents contain ``product_query``."""
    state = ASSOCIATION_STATE
    if state["rules"] is None:
        return pd.DataFrame({"info": ["⚠️ No rules loaded."]})
    if not product_query:
        return state["rules"].head(top_n)

    q = product_query.upper()
    rules = state["rules"]

    def contains(s) -> bool:
        return any(q in item.upper() for item in s)

    mask = rules["antecedents"].apply(contains) | rules["consequents"].apply(contains)
    matched = rules[mask].head(top_n).copy()

    # Make readable
    matched["antecedents"] = matched["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    matched["consequents"] = matched["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return matched[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)


def build_association_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "### Rule Search\n"
            "Type part of a product name (e.g., `BAG`, `HEART`, `MUG`). The "
            "table below shows the top association rules involving that product."
        )

        query = gr.Textbox(label="Product keyword", value="")
        top_n = gr.Slider(5, 30, value=10, step=1, label="Number of rules to show")
        run_btn = gr.Button("🔎 Search")
        results_df = gr.Dataframe(headers=["antecedents", "consequents", "support", "confidence", "lift"])
        run_btn.click(search_rules, inputs=[query, top_n], outputs=results_df)

    return tab


# ---------------------------------------------------------------------------
# Main app assembly
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble the 4-tab application."""
    with gr.Blocks(title="CSC 588 — Data Mining Demo", theme="soft") as app:
        gr.Markdown("# 🧠 CSC 588 — Data Mining Project Demo")

        with gr.Tabs():
            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT_MARKDOWN)
            with gr.Tab("🔮 Classification"):
                build_classification_tab()
            with gr.Tab("🎯 Clustering"):
                build_clustering_tab()
            with gr.Tab("🛒 Association"):
                build_association_tab()

    return app


def launch_app(
    classification_models: dict[str, Any],
    classification_preprocessor: Any,
    classification_feature_template: pd.DataFrame,
    clustering_models: dict[str, np.ndarray],
    clustering_X_scaled: np.ndarray,
    clustering_feature_names: list[str],
    clustering_scaler: Any,
    association_rules_df: pd.DataFrame,
    share: bool = True,
) -> None:
    """Wire all trained artefacts into the UI globals and launch the app."""
    CLASSIFICATION_STATE["models"] = classification_models
    CLASSIFICATION_STATE["preprocessor"] = classification_preprocessor
    CLASSIFICATION_STATE["feature_template"] = classification_feature_template

    CLUSTERING_STATE["models"] = clustering_models
    CLUSTERING_STATE["X_scaled"] = clustering_X_scaled
    CLUSTERING_STATE["feature_names"] = clustering_feature_names
    CLUSTERING_STATE["scaler"] = clustering_scaler

    ASSOCIATION_STATE["rules"] = association_rules_df

    app = build_app()
    app.launch(share=share, debug=False)
