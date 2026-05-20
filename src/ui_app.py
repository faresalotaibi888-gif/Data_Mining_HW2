"""Business-framed Gradio UI for the CSC 588 Data Mining project.

Each tab answers a specific business question rather than presenting raw
algorithm output. The design is deliberately problem-first to satisfy
report instruction (c): clarify why the study was undertaken and what
hypotheses were tested.

Tabs (in order):
    1. About — problem framing, hypotheses, key findings
    2. Will this visitor buy?         (Classification, Online Shoppers)
    3. Who are my customers?          (Clustering, CC General)
    4. What products go together?     (Association, Online Retail II)
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level state (populated by launch_app)
# ---------------------------------------------------------------------------

CLASSIFICATION_STATE: dict[str, Any] = {
    "models": None, "preprocessor": None, "feature_template": None,
}
CLUSTERING_STATE: dict[str, Any] = {
    "models": None, "X_scaled": None, "feature_names": None, "scaler": None,
}
ASSOCIATION_STATE: dict[str, Any] = {"rules": None}


# ---------------------------------------------------------------------------
# About tab — the "why" of the project
# ---------------------------------------------------------------------------

ABOUT_MARKDOWN = """
# 📊 Data Mining for E-Commerce — Three Questions, Three Algorithms

This project applies the three foundational paradigms of data mining to **real business problems
faced by online retailers every day**.

## Why this study was undertaken

E-commerce companies generate enormous volumes of behavioural and transactional data — yet
most of it is never used to inform decisions. We focus on **three concrete business questions**
that data mining can answer at scale:

| Business question | Paradigm | Dataset |
|---|---|---|
| **Will this visitor make a purchase?** | Classification | Online Shoppers Purchasing Intention (UCI) |
| **Who are my customer segments?** | Clustering | Credit Card Customers (CC General, Kaggle) |
| **Which products are bought together?** | Association rule mining | Online Retail II (UCI) |

## The hypotheses we tested

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Ensemble methods outperform single-model methods | ✅ **Supported** |
| H2 | Tree ensembles offer the best speed/accuracy trade-off | ✅ **Supported** |
| H3 | SMOTE oversampling improves minority-class recall | ✅ **Supported** |
| H4 | DBSCAN reveals non-spherical structure missed by K-Means | ❌ **Refuted** |
| H5 | FP-Growth is substantially faster than Apriori | ❌ **Refuted** |
| H6 | Hyperparameter tuning beats default parameters | ✅ **Supported** |

Two hypotheses were refuted — those are honest empirical findings, not failures. They show that
algorithm choice should match data geometry and dataset size, not textbook expectations.

## Key findings at a glance

| 🏆 Best classifier | 🎯 Cleanest segmentation | 🛒 Strongest rule |
|---|---|---|
| **AdaBoost** F1 = 0.668 | **K-Means k=4**, silhouette 0.198 | Green Teacup ↔ Roses Teacup, **lift 17.7** |

## How to use this demo

Use the tabs on the left in order:

1. **Will this visitor buy?** — try the classifier on a synthetic shopper
2. **Who are my customers?** — see which segment a customer falls into
3. **What products go together?** — explore co-purchase rules

Each tab translates the algorithm output into a **business recommendation** — not a raw metric.
"""


# ===========================================================================
# Tab 1 — Will this visitor buy? (Classification)
# ===========================================================================

CLASSIFICATION_INTRO = """
## 💼 Business question

> *"A visitor is browsing our online store right now. Should we spend retargeting
> ad budget on them?"*

Only **~15% of visitors actually purchase**. Spending retargeting ad budget on the
85% who won't convert wastes money. This classifier predicts the probability that
a session will end in a purchase, so the marketing team can prioritise the visitors
most worth retargeting.

**Try it below**: enter a visitor's behaviour, click "Predict", and the model tells
you whether to invest in them — and which behaviours drove the prediction.
"""


def _build_classification_input(
    pages_admin, time_admin,
    pages_info, time_info,
    pages_product, time_product,
    bounce_rate, exit_rate, page_value, special_day,
    month, weekend, visitor_type,
    operating_systems, browser, region, traffic_type,
):
    """Assemble a one-row DataFrame matching the training schema."""
    template = CLASSIFICATION_STATE["feature_template"]
    if template is None:
        raise gr.Error("Classification UI is not initialised.")

    row = template.copy()
    row["Administrative"] = int(pages_admin)
    row["Administrative_Duration"] = float(time_admin)
    row["Informational"] = int(pages_info)
    row["Informational_Duration"] = float(time_info)
    row["ProductRelated"] = int(pages_product)
    row["ProductRelated_Duration"] = float(time_product)
    row["BounceRates"] = float(bounce_rate)
    row["ExitRates"] = float(exit_rate)
    row["PageValues"] = float(page_value)
    row["SpecialDay"] = float(special_day)
    row["Month"] = month
    row["Weekend"] = bool(weekend)
    row["VisitorType"] = visitor_type
    row["OperatingSystems"] = int(operating_systems)
    row["Browser"] = int(browser)
    row["Region"] = int(region)
    row["TrafficType"] = int(traffic_type)
    return row


def _predict_purchase(
    pages_admin, time_admin,
    pages_info, time_info,
    pages_product, time_product,
    bounce_rate, exit_rate, page_value, special_day,
    month, weekend, visitor_type,
    operating_systems, browser, region, traffic_type,
    model_name,
):
    """Run prediction and return business-framed outputs."""
    state = CLASSIFICATION_STATE
    if state["models"] is None:
        return "_Models not loaded_", "_n/a_", "_n/a_", None

    model = state["models"].get(model_name)
    if model is None:
        return f"Model '{model_name}' not available.", "_n/a_", "_n/a_", None

    row = _build_classification_input(
        pages_admin, time_admin, pages_info, time_info,
        pages_product, time_product, bounce_rate, exit_rate,
        page_value, special_day, month, weekend, visitor_type,
        operating_systems, browser, region, traffic_type,
    )
    X = state["preprocessor"].transform(row)

    # Probability if available; else decision function or hard label
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        d = float(model.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-d))
    else:
        proba = float(model.predict(X)[0])

    # Business framing
    if proba >= 0.5:
        verdict_md = (
            f"### 🟢 HIGH-INTENT VISITOR — worth retargeting\n\n"
            f"**Predicted purchase probability: {proba:.1%}**\n\n"
            f"_This visitor is significantly more likely to convert than the average shopper "
            f"(~15% base rate). Recommended action: include in retargeting and remarketing campaigns._"
        )
    elif proba >= 0.25:
        verdict_md = (
            f"### 🟡 MEDIUM-INTENT VISITOR — consider light touch\n\n"
            f"**Predicted purchase probability: {proba:.1%}**\n\n"
            f"_Above the base rate but not strongly committed. Recommended action: low-cost "
            f"engagement (email newsletter), not premium ad spend._"
        )
    else:
        verdict_md = (
            f"### 🔴 LOW-INTENT VISITOR — skip retargeting\n\n"
            f"**Predicted purchase probability: {proba:.1%}**\n\n"
            f"_Below the base rate. Recommended action: do not spend retargeting budget on this "
            f"profile; concentrate resources on higher-intent traffic._"
        )

    # Behavioural insight — what drove the prediction (intuitive narrative)
    drivers_md = "### What drove this prediction?\n\n"
    drivers = []
    if page_value > 5:
        drivers.append(f"- 💎 **High Page Value ({page_value:.1f})** — this is the single "
                       f"strongest signal in our dataset.")
    elif page_value > 0:
        drivers.append(f"- 💰 **Some Page Value ({page_value:.1f})** — moderately positive.")
    else:
        drivers.append("- 📉 **Zero Page Value** — the model's strongest predictor is missing here.")

    if pages_product >= 20:
        drivers.append(f"- 🛍️ **{int(pages_product)} product page views** — strong intent signal.")
    elif pages_product >= 5:
        drivers.append(f"- 🛍️ **{int(pages_product)} product page views** — moderate engagement.")
    else:
        drivers.append(f"- 🚶 **Only {int(pages_product)} product page views** — low engagement.")

    if bounce_rate > 0.1:
        drivers.append(f"- ⚠️ **High bounce rate ({bounce_rate:.2f})** — visitor leaves quickly.")
    if exit_rate > 0.1:
        drivers.append(f"- ⚠️ **High exit rate ({exit_rate:.2f})** — pages aren't holding attention.")

    if visitor_type == "Returning_Visitor":
        drivers.append("- 🔁 **Returning visitor** — shows pre-existing interest.")
    elif visitor_type == "New_Visitor":
        drivers.append("- ✨ **New visitor** — first-time exposure to the brand.")

    if special_day > 0.5:
        drivers.append(f"- 🎁 **Near a special day** (proximity {special_day:.1f}) — "
                       f"gift-shopping season boosts intent.")

    drivers_md += "\n".join(drivers)

    # Stat summary box for the right column
    stats_md = (
        f"### Visitor profile summary\n\n"
        f"- **Product pages viewed:** {int(pages_product)}\n"
        f"- **Time on product pages:** {time_product:.0f} seconds\n"
        f"- **Page value:** {page_value:.2f}\n"
        f"- **Bounce rate:** {bounce_rate:.3f}\n"
        f"- **Exit rate:** {exit_rate:.3f}\n"
        f"- **Visitor type:** {visitor_type}\n"
        f"- **Day:** {'Weekend' if weekend else 'Weekday'}, {month}\n"
        f"- **Predicted by:** {model_name}"
    )

    # Probability for the gauge display
    return verdict_md, stats_md, drivers_md, proba


def build_classification_tab():
    """Tab 2 — Will this visitor buy?"""
    with gr.Column() as tab:
        gr.Markdown(CLASSIFICATION_INTRO)

        gr.Markdown("### 🧮 Enter the visitor's behaviour")

        with gr.Row():
            # Left column: behaviour inputs
            with gr.Column(scale=1):
                gr.Markdown("**Page interaction**")
                pages_admin = gr.Number(value=0, label="Administrative pages viewed (login, account)")
                time_admin = gr.Number(value=0, label="Time on administrative pages (seconds)")
                pages_info = gr.Number(value=0, label="Informational pages viewed (FAQ, about)")
                time_info = gr.Number(value=0, label="Time on informational pages (seconds)")
                pages_product = gr.Number(value=10, label="Product pages viewed")
                time_product = gr.Number(value=300, label="Time on product pages (seconds)")

            with gr.Column(scale=1):
                gr.Markdown("**Engagement metrics**")
                bounce_rate = gr.Slider(0, 0.2, value=0.02, step=0.005,
                                        label="Average bounce rate (Google Analytics)")
                exit_rate = gr.Slider(0, 0.2, value=0.04, step=0.005,
                                      label="Average exit rate (Google Analytics)")
                page_value = gr.Slider(0, 50, value=5.0, step=0.5,
                                       label="Page Value (estimated $ revenue per page)")
                special_day = gr.Slider(0, 1, value=0.0, step=0.1,
                                        label="Closeness to a special day (0=none, 1=on the day)")

                gr.Markdown("**Context**")
                month = gr.Dropdown(
                    ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    value="Nov", label="Month")
                weekend = gr.Checkbox(value=False, label="Weekend visit?")
                visitor_type = gr.Dropdown(
                    ["Returning_Visitor", "New_Visitor", "Other"],
                    value="Returning_Visitor", label="Visitor type")

        with gr.Accordion("Advanced — technical attributes (anonymised codes)", open=False):
            with gr.Row():
                operating_systems = gr.Number(value=2, precision=0, label="Operating system code")
                browser = gr.Number(value=2, precision=0, label="Browser code")
                region = gr.Number(value=1, precision=0, label="Region code")
                traffic_type = gr.Number(value=2, precision=0, label="Traffic source code")

        # Read available model names from state at build time (state is
        # populated by launch_app BEFORE build_app runs).
        _model_names = list((CLASSIFICATION_STATE.get("models") or {}).keys())
        _default_model = "AdaBoost" if "AdaBoost" in _model_names else (
            _model_names[0] if _model_names else None
        )

        with gr.Row():
            model_name = gr.Dropdown(
                choices=_model_names,
                value=_default_model,
                label="Choose model (default: AdaBoost — our top performer)",
                interactive=True,
            )
            predict_btn = gr.Button("🔮 Predict purchase probability", variant="primary", scale=2)

        gr.Markdown("---")
        gr.Markdown("## 📈 Prediction result")

        with gr.Row():
            with gr.Column(scale=2):
                verdict_box = gr.Markdown("_Click 'Predict' to see the result._")
                drivers_box = gr.Markdown("")
            with gr.Column(scale=1):
                stats_box = gr.Markdown("")
                prob_number = gr.Number(label="Probability of purchase", precision=4, interactive=False)

        predict_btn.click(
            _predict_purchase,
            inputs=[
                pages_admin, time_admin, pages_info, time_info,
                pages_product, time_product, bounce_rate, exit_rate,
                page_value, special_day, month, weekend, visitor_type,
                operating_systems, browser, region, traffic_type,
                model_name,
            ],
            outputs=[verdict_box, stats_box, drivers_box, prob_number],
        )

    return tab


# ===========================================================================
# Tab 3 — Who are my customers? (Clustering)
# ===========================================================================

CLUSTERING_INTRO = """
## 💼 Business question

> *"We manage 8,950 credit card customers. We can't design 8,950 different campaigns —
> but we can design 4. Who are our segments?"*

K-Means produced four interpretable segments on the CC General dataset. Use the form
below to see **which segment a customer belongs to**, what they look like, and what
marketing action makes sense for them.
"""

# Persona definitions — built once at launch from the K-Means profile table
CLUSTER_PERSONAS = {
    0: {
        "name": "Low-engagement majority",
        "share": "~44%",
        "icon": "🚶",
        "traits": "Low balance ($1,013) · Low monthly purchases ($270) · Low credit limit ($3,279)",
        "action": "**Recommended action:** Activation campaigns — onboarding emails, "
                  "introductory promotions, gentle reactivation offers.",
    },
    1: {
        "name": "Premium spenders",
        "share": "~5%",
        "icon": "💎",
        "traits": "High balance ($3,551) · Very high purchases ($7,682) · High credit limit ($9,697)",
        "action": "**Recommended action:** Premium retention — exclusive perks, "
                  "concierge service, premium card upgrade offers.",
    },
    2: {
        "name": "Cash-advance heavy",
        "share": "~13%",
        "icon": "💵",
        "traits": "Highest balance ($4,602) · Low purchases ($502) · Very high cash advance ($4,522)",
        "action": "**Recommended action:** Financial wellness — debt consolidation offers, "
                  "fee-reduction promotions, balance transfer products.",
    },
    3: {
        "name": "Engaged casual buyers",
        "share": "~38%",
        "icon": "🛍️",
        "traits": "Low balance ($895) · Moderate purchases ($1,236) · Frequent installments",
        "action": "**Recommended action:** Loyalty programs — cashback rewards, "
                  "installment-plan promotions, category-based offers.",
    },
}


def _assign_cluster(balance, purchases, oneoff, installments,
                    cash_advance, purchase_freq, credit_limit, payments, tenure):
    """Assign a new customer to the closest K-Means cluster centroid (Euclidean in scaled space)."""
    state = CLUSTERING_STATE
    if state["X_scaled"] is None or state["scaler"] is None:
        return "_Clustering UI is not initialised._", ""

    scaler = state["scaler"]
    feature_names = state["feature_names"]
    X_scaled = state["X_scaled"]
    labels = state["models"].get("K-Means")
    if labels is None:
        return "_K-Means labels not loaded._", ""

    # Build the input row in the feature_names order with sensible defaults for unspecified columns
    user_values = {
        "BALANCE": float(balance),
        "PURCHASES": float(purchases),
        "ONEOFF_PURCHASES": float(oneoff),
        "INSTALLMENTS_PURCHASES": float(installments),
        "CASH_ADVANCE": float(cash_advance),
        "PURCHASES_FREQUENCY": float(purchase_freq),
        "CREDIT_LIMIT": float(credit_limit),
        "PAYMENTS": float(payments),
        "TENURE": float(tenure),
    }
    # Median fill anything not given
    medians = np.median(scaler.inverse_transform(X_scaled), axis=0)
    row_values = []
    for i, col in enumerate(feature_names):
        if col in user_values:
            row_values.append(user_values[col])
        else:
            row_values.append(medians[i])
    row = np.array(row_values).reshape(1, -1)
    row_scaled = scaler.transform(row)

    # Compute centroids from labels
    unique = sorted(set(labels))
    centroids = np.vstack([X_scaled[labels == c].mean(axis=0) for c in unique])
    distances = np.linalg.norm(centroids - row_scaled, axis=1)
    closest = int(unique[int(np.argmin(distances))])

    persona = CLUSTER_PERSONAS.get(closest, {
        "name": f"Cluster {closest}", "share": "n/a", "icon": "❓",
        "traits": "Custom segment", "action": "",
    })

    result_md = (
        f"# {persona['icon']} Cluster {closest} — {persona['name']}\n\n"
        f"**Population share:** {persona['share']} of the 8,950 customers in this database.\n\n"
        f"### Typical traits of this segment\n"
        f"{persona['traits']}\n\n"
        f"### Marketing recommendation\n"
        f"{persona['action']}"
    )

    # Build the persona gallery
    gallery_md = "### 🎨 All four segments at a glance\n\n"
    for c, p in CLUSTER_PERSONAS.items():
        highlight = " ← **your customer**" if c == closest else ""
        gallery_md += f"- {p['icon']} **Cluster {c} — {p['name']}** ({p['share']}){highlight}\n"

    return result_md, gallery_md


def build_clustering_tab():
    """Tab 3 — Who are my customers?"""
    with gr.Column() as tab:
        gr.Markdown(CLUSTERING_INTRO)

        gr.Markdown("### 🧮 Enter a customer's profile")

        with gr.Row():
            with gr.Column():
                balance = gr.Number(value=1500, label="Monthly balance ($)")
                purchases = gr.Number(value=800, label="Monthly purchases ($)")
                oneoff = gr.Number(value=300, label="One-off purchases ($)")
                installments = gr.Number(value=500, label="Installment purchases ($)")
            with gr.Column():
                cash_advance = gr.Number(value=200, label="Cash advances ($)")
                purchase_freq = gr.Slider(0, 1, value=0.5, step=0.05,
                                          label="Purchase frequency (0–1)")
                credit_limit = gr.Number(value=5000, label="Credit limit ($)")
                payments = gr.Number(value=2000, label="Monthly payments ($)")
                tenure = gr.Slider(6, 12, value=12, step=1, label="Account tenure (months)")

        assign_btn = gr.Button("🎯 Find this customer's segment", variant="primary")

        gr.Markdown("---")
        result_box = gr.Markdown("_Enter values above and click 'Find this customer's segment'._")
        gr.Markdown("---")
        gallery_box = gr.Markdown("")

        assign_btn.click(
            _assign_cluster,
            inputs=[balance, purchases, oneoff, installments,
                    cash_advance, purchase_freq, credit_limit, payments, tenure],
            outputs=[result_box, gallery_box],
        )
    return tab


# ===========================================================================
# Tab 4 — What products go together? (Association)
# ===========================================================================

ASSOCIATION_INTRO = """
## 💼 Business question

> *"I'm planning the homepage layout. Which products should I bundle? Which co-purchase
> patterns will generate the most cross-sell revenue?"*

We mined 19,500 UK invoices with Apriori and FP-Growth. Each rule tells you: when a
customer buys product A, **how much more likely than chance** are they to also buy B?
That ratio is called **lift** — a lift of 10 means "10× more likely than random chance".

Use the search below to find rules involving specific products. The recommendations
panel shows what to do with each rule.
"""


def _lift_in_words(lift):
    """Translate a lift number into business intuition."""
    if lift >= 10:
        return f"🔥 **Extremely strong** — {lift:.1f}× more likely than random chance"
    if lift >= 5:
        return f"⭐ **Strong** — {lift:.1f}× more likely than random chance"
    if lift >= 2:
        return f"📈 **Notable** — {lift:.1f}× more likely than random chance"
    return f"➡️ **Modest** — {lift:.1f}× more likely than random chance"


def _format_itemset(s):
    if isinstance(s, str):
        return s
    try:
        return ", ".join(sorted(s))
    except TypeError:
        return str(s)


def _search_rules(query, top_n):
    rules_df = ASSOCIATION_STATE["rules"]
    if rules_df is None or len(rules_df) == 0:
        return "_No rules loaded._", pd.DataFrame()

    df = rules_df.copy()
    # Normalise frozensets to readable strings
    df["A"] = df["antecedents"].apply(_format_itemset)
    df["B"] = df["consequents"].apply(_format_itemset)

    query = (query or "").strip().lower()
    if query:
        mask = df["A"].str.lower().str.contains(query) | df["B"].str.lower().str.contains(query)
        df = df[mask]

    if len(df) == 0:
        return (f"_No rules found matching **{query}**. Try a more generic term like 'BAG' "
                f"or 'TEACUP'._"), pd.DataFrame()

    df = df.sort_values("lift", ascending=False).head(int(top_n))

    # Build a business-framed narrative panel for the top rule
    top = df.iloc[0]
    narrative = (
        f"## 🏆 Strongest match\n\n"
        f"### Customers who buy **{top['A']}**\n"
        f"### often also buy **{top['B']}**\n\n"
        f"- {_lift_in_words(top['lift'])}\n"
        f"- **Confidence:** {top['confidence']:.0%} of customers who buy the first product also buy the second.\n"
        f"- **Support:** {top['support']:.1%} of all invoices contain this combination.\n\n"
        f"### 💡 What to do with this insight\n\n"
        f"- **Bundle promotion:** offer the two together at a small discount.\n"
        f"- **Cross-sell prompt:** when a customer adds **{top['A']}** to their cart, "
        f"recommend **{top['B']}** at checkout.\n"
        f"- **Store layout:** place these items adjacent in physical stores or on the homepage."
    )

    # Build a clean table for display
    display = df[["A", "B", "support", "confidence", "lift"]].copy()
    display.columns = ["When they buy", "They also buy", "Support", "Confidence", "Lift"]
    display["Support"] = display["Support"].apply(lambda x: f"{x:.1%}")
    display["Confidence"] = display["Confidence"].apply(lambda x: f"{x:.0%}")
    display["Lift"] = display["Lift"].apply(lambda x: f"{x:.1f}×")

    return narrative, display


def build_association_tab():
    """Tab 4 — What products go together?"""
    with gr.Column():
        gr.Markdown(ASSOCIATION_INTRO)

        gr.Markdown("### 🔎 Search rules by product keyword")
        with gr.Row():
            query = gr.Textbox(value="", label="Product keyword (e.g., 'TEACUP', 'BAG', 'CHRISTMAS')")
            top_n = gr.Slider(5, 30, value=10, step=1, label="Number of rules to display")
            search_btn = gr.Button("🔍 Find matching rules", variant="primary")

        gr.Markdown("---")
        narrative_box = gr.Markdown("_Enter a keyword above and click 'Find matching rules'._")
        gr.Markdown("---")
        gr.Markdown("### 📋 Detailed rules table")
        rules_table = gr.Dataframe(
            headers=["When they buy", "They also buy", "Support", "Confidence", "Lift"],
            interactive=False,
        )

        search_btn.click(_search_rules, inputs=[query, top_n], outputs=[narrative_box, rules_table])


# ===========================================================================
# Main app assembly
# ===========================================================================

def build_app() -> gr.Blocks:
    """Assemble the 4-tab application with business-first framing."""
    with gr.Blocks(
        title="CSC 588 — Data Mining for E-Commerce",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="amber"),
    ) as app:
        gr.Markdown(
            "# 🛒 Data Mining for E-Commerce\n"
            "_CSC 588 Data Warehousing & Mining — Project Demo_"
        )

        with gr.Tabs():
            with gr.Tab("📊 About this project"):
                gr.Markdown(ABOUT_MARKDOWN)

            with gr.Tab("🔮 Will this visitor buy?"):
                build_classification_tab()

            with gr.Tab("👥 Who are my customers?"):
                build_clustering_tab()

            with gr.Tab("🛍️ What products go together?"):
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
    """Wire all trained artefacts into the UI globals and launch the app.

    Signature is unchanged from the previous version so demo.ipynb keeps
    working — only the UI's framing and content has been rewritten.
    """
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
