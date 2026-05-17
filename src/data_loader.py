"""Dataset loaders for the CSC 588 Data Mining project.

This module provides a single point of entry for the three datasets used
across the classification, clustering, and association-rule-mining tracks.
All loaders cache to ``data/`` so subsequent calls are instantaneous.

Datasets
--------
1. Online Shoppers Purchasing Intention  (UCI)  — classification
2. Credit Card Customers (CC General)     (Kaggle) — clustering
3. Online Retail II                       (UCI)  — association mining
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and remote sources
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ONLINE_SHOPPERS_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00468/online_shoppers_intention.csv"
)
ONLINE_RETAIL_II_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00502/online_retail_II.xlsx"
)

# CC General is hosted on Kaggle which requires authentication. The user
# should download the CSV once and place it in data/. The notebook for the
# clustering track includes a Colab cell that handles this with kaggle.json.
CC_GENERAL_FILENAME = "CC_GENERAL.csv"
CC_GENERAL_KAGGLE_URL = "https://www.kaggle.com/datasets/arjunbhasin2013/ccdata"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_if_missing(url: str, local_path: Path) -> Path:
    """Download ``url`` to ``local_path`` only if the file does not yet exist."""
    local_path = Path(local_path)
    if local_path.exists():
        return local_path

    print(f"[data_loader] Downloading {url}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"[data_loader] Cached at {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_online_shoppers() -> pd.DataFrame:
    """Load the Online Shoppers Purchasing Intention dataset.

    Returns
    -------
    pd.DataFrame
        12,330 rows × 18 columns. The target column is ``Revenue``
        (bool: True if the session ended in a purchase).
    """
    path = DATA_DIR / "online_shoppers_intention.csv"
    _download_if_missing(ONLINE_SHOPPERS_URL, path)
    return pd.read_csv(path)


def load_cc_general() -> pd.DataFrame:
    """Load the Credit Card Customers (CC General) dataset for clustering.

    The file must be placed at ``data/CC_GENERAL.csv``. Inside the
    clustering notebook a cell guides the Colab user through uploading
    ``kaggle.json`` and downloading via the Kaggle CLI.

    Returns
    -------
    pd.DataFrame
        8,950 rows × 18 columns of credit card usage behaviour.

    Raises
    ------
    FileNotFoundError
        If the CSV has not been placed in ``data/``.
    """
    path = DATA_DIR / CC_GENERAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path}. Download CC_GENERAL.csv from "
            f"{CC_GENERAL_KAGGLE_URL} and place it in the data/ folder, "
            "or run the Kaggle download cell at the top of "
            "notebooks/02_clustering.ipynb."
        )
    return pd.read_csv(path)


def load_online_retail_ii(year: str = "2010-2011") -> pd.DataFrame:
    """Load the Online Retail II transactional dataset.

    Parameters
    ----------
    year : {"2010-2011", "2009-2010", "both"}, default "2010-2011"
        Which sheet of the workbook to load. ``"both"`` concatenates
        the two sheets (≈ 1M rows total).

    Returns
    -------
    pd.DataFrame
        Raw transaction lines with columns Invoice, StockCode,
        Description, Quantity, InvoiceDate, Price, Customer ID, Country.
    """
    path = DATA_DIR / "online_retail_II.xlsx"
    _download_if_missing(ONLINE_RETAIL_II_URL, path)

    if year == "both":
        df1 = pd.read_excel(path, sheet_name="Year 2009-2010")
        df2 = pd.read_excel(path, sheet_name="Year 2010-2011")
        return pd.concat([df1, df2], ignore_index=True)
    if year == "2009-2010":
        return pd.read_excel(path, sheet_name="Year 2009-2010")
    if year == "2010-2011":
        return pd.read_excel(path, sheet_name="Year 2010-2011")
    raise ValueError(
        f"year must be one of '2010-2011', '2009-2010', 'both'; got {year!r}"
    )


# ---------------------------------------------------------------------------
# Convenience reporting
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print a compact textual summary of a dataframe."""
    print(f"\n=== {name} ===")
    print(f"Shape : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nMissing values:\n{missing}")
    else:
        print("\nMissing values: none")
    print(f"\nHead:\n{df.head(3)}")


if __name__ == "__main__":
    # Smoke test — run with: python -m src.data_loader
    shoppers = load_online_shoppers()
    dataset_summary(shoppers, "Online Shoppers Purchasing Intention")
