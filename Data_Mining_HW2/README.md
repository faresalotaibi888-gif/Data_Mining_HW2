# CSC 588 — Data Mining Course Project

A unified data mining benchmark covering the three major paradigms — **classification**, **clustering**, and **association rule mining** — implemented with scikit-learn and demonstrated through an interactive Gradio UI.

> **Course:** CSC 588 Data Warehousing & Mining
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

## 🧪 Algorithms

**Classification (10):** Logistic Regression · KNN · Decision Tree · Random Forest · SVM (RBF) · Gaussian Naive Bayes · Linear Discriminant Analysis · Gradient Boosting · AdaBoost · Multi-Layer Perceptron

**Clustering (3):** K-Means · Agglomerative Hierarchical · DBSCAN

**Association (2):** Apriori · FP-Growth

## 🚀 Quickstart (Colab)

Open any notebook directly in Colab:

| Notebook | Open in Colab |
|---|---|
| `01_classification.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/01_classification.ipynb) |
| `02_clustering.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/02_clustering.ipynb) |
| `03_association.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/03_association.ipynb) |
| `04_ui_demo.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faresalotaibi888-gif/Data_Mining_HW2/blob/main/notebooks/04_ui_demo.ipynb) |

Each notebook's first cell clones this repo and installs dependencies:

```python
!git clone https://github.com/faresalotaibi888-gif/Data_Mining_HW2.git
%cd Data_Mining_HW2
!pip install -r requirements.txt -q
```

## 📁 Repository Structure

```
Data_Mining_HW2/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/              # Colab-ready demonstration notebooks
│   ├── 01_classification.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_association.ipynb
│   └── 04_ui_demo.ipynb
├── src/                    # Reproducible Python modules
│   ├── data_loader.py      # Dataset loaders with auto-download
│   ├── preprocessing.py    # Pipelines for each track
│   ├── classification.py   # 10 classifiers + GridSearchCV
│   ├── clustering.py       # 3 clusterers + diagnostics
│   ├── association.py      # Apriori + FP-Growth
│   ├── evaluation.py       # Metric tables and plots
│   └── ui_app.py           # Gradio interface
├── data/                   # Raw datasets (auto-populated)
├── results/                # Generated tables and figures
├── screenshots/            # UI and evaluation screenshots
├── report/                 # Word report (.docx)
└── presentation/           # PPT slides (.pptx)
```

## 🔬 Validation Strategy

- **Classification:** Stratified 5-fold cross-validation with `GridSearchCV` for tuning, plus a held-out test set for final reporting. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, train/inference time.
- **Clustering:** Silhouette score, Davies-Bouldin Index, Calinski-Harabasz; elbow method and dendrogram for *k* selection.
- **Association:** Support, Confidence, Lift, Conviction; runtime comparison Apriori vs. FP-Growth.

## 📜 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

Fares Alotaibi · CSC 588 · Fall 2025
