# CSC 588 — Data Mining Course Project

A unified data mining benchmark covering the three major paradigms — **classification**, **clustering**, and **association rule mining** — implemented with scikit-learn and demonstrated through an interactive Gradio UI.

> **Course:** CSC 588 Data Warehousing & Mining
> **Author:** Fares Alotaibi
> **Platform:** Google Colab (zero local setup required)

---

## 🎯 Objectives

1. Benchmark **10 classification algorithms** on a real-world e-commerce purchase intent dataset.
2. Compare **3 clustering algorithms** on high-dimensional customer financial behavior data.
3. Apply **2 association rule mining algorithms** to a large transactional retail dataset.
4. Tune hyperparameters via cross-validated grid search and quantify accuracy/runtime trade-offs.
5. Compare results against published literature on the same datasets.

## 📊 Datasets

| Track | Dataset | Source | Shape |
|---|---|---|---|
| Classification | Online Shoppers Purchasing Intention | [UCI ML Repo](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | 12,330 × 18 |
| Clustering | Credit Card Customers (CC General) | [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) | 8,950 × 18 |
| Association | Online Retail II | [UCI ML Repo](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | ~1M transactions |

> **Self-contained**: `CC_GENERAL.csv` is bundled in `data/`. The two UCI datasets auto-download on first run.

## 🧪 Algorithms

**Classification (10):** Logistic Regression · KNN · Decision Tree · Random Forest · SVM (RBF) · Gaussian Naive Bayes · Linear Discriminant Analysis · Gradient Boosting · AdaBoost · Multi-Layer Perceptron

**Clustering (3):** K-Means · Agglomerative Hierarchical (Ward) · DBSCAN

**Association (2):** Apriori · FP-Growth

## 🚀 Quickstart — Run the entire project in Colab

Open the single demo notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/demo.ipynb)

Or paste this URL in your browser:

```
https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/demo.ipynb
```

Then: `Runtime → Run all`. The notebook clones the repo, installs dependencies, and runs all three tracks plus the UI end-to-end (~20 minutes).

### Single notebook covers everything

`demo.ipynb` is structured in 7 sections:

| Section | Contents | Runtime |
|---|---|---|
| **A. Setup** | Clone repo, install dependencies, import modules | ~1 min |
| **B. Data Exploration** | Inspect the Online Shoppers dataset, plot class balance | ~10 sec |
| **C. Classification** | Preprocess → 10 algorithms × GridSearchCV → 4 result figures | 8–15 min |
| **D. Clustering** | Diagnostics → 3 algorithms → PCA visualisation → cluster profiles | 2–3 min |
| **E. Association** | Reshape baskets → Apriori vs FP-Growth → top rules | 1 min |
| **F. UI** | Launch Gradio app with public share URL | 30 sec |
| **G. Hypothesis Evaluation** | Verdicts for all six pre-registered hypotheses | (markdown) |

## 📁 Repository Structure

```
Data_Mining_HW2/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── demo.ipynb          ← single consolidated notebook
├── src/                    ← reusable Python modules
│   ├── data_loader.py      ← loads all three datasets
│   ├── preprocessing.py    ← pipelines for each track
│   ├── classification.py   ← 10 classifiers + GridSearchCV
│   ├── clustering.py       ← K-Means, Agglo, DBSCAN + diagnostics
│   ├── association.py      ← Apriori + FP-Growth
│   ├── evaluation.py       ← shared plot/table helpers
│   └── ui_app.py           ← Gradio interface
├── data/
│   └── CC_GENERAL.csv      ← bundled (others auto-download)
└── results/                ← populated when the notebook runs
    ├── figures/
    └── tables/
```

## 🔬 Validation Strategy

- **Classification:** Stratified 5-fold cross-validation with `GridSearchCV` for tuning; held-out test set for final reporting. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, train time, inference time.
- **Clustering:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz index; elbow method, k-distance plot, and dendrogram for parameter selection.
- **Association:** Support, Confidence, Lift, Conviction; runtime comparison Apriori vs FP-Growth.

## 🧱 Hyperparameter Tuning

Every classifier is tuned via `GridSearchCV` with stratified 5-fold CV. The grids:

| Algorithm | Parameters tuned | Grid values |
|---|---|---|
| Logistic Regression | `C` | {0.1, 1.0, 10.0} |
| K-Nearest Neighbors | `n_neighbors`, `weights` | {5, 11, 21} × {uniform, distance} |
| Decision Tree | `max_depth`, `min_samples_split` | {5, 10, None} × {2, 10} |
| Random Forest | `n_estimators`, `max_depth` | {100} × {10, None} |
| SVM (RBF) | `C`, `gamma` | {1.0} × {scale} |
| Gaussian Naive Bayes | `var_smoothing` | {1e-9, 1e-8, 1e-7} |
| Linear Discriminant Analysis | `solver` | {svd} |
| Gradient Boosting | `n_estimators`, `max_depth` | {100} × {3, 5} |
| AdaBoost | `n_estimators`, `learning_rate` | {50, 100} × {0.5, 1.0} |
| MLP | `hidden_layer_sizes` | {(50,), (100,)} |

Best parameters per algorithm are saved in `results/tables/classification_results.csv` and printed in Section C of the demo notebook.

## 📦 Dependencies

All pinned in `requirements.txt`. The notebook installs everything automatically:
- scikit-learn ≥ 1.3, imbalanced-learn ≥ 0.11
- mlxtend ≥ 0.23 (association mining)
- matplotlib, seaborn, plotly
- pandas, numpy, scipy
- gradio ≥ 4.0 (UI)

## 📜 License

MIT — see [LICENSE](LICENSE).
