"""
data_loader.py — Dataset acquisition and loading.

Handles three datasets:
    1. Online Shoppers Purchasing Intention (UCI)     -- classification
    2. CC General Credit Card Customers (Kaggle)      -- clustering
    3. Online Retail II (UCI)                         -- association

Each loader function returns a clean pandas DataFrame. Datasets are cached
locally in DATA_DIR after first download.

Colab-friendly: all loaders handle missing data gracefully and provide
instructions when manual upload is required (Kaggle dataset).

Usage
-----
>>> from src.data_loader import (
...     load_online_shoppers,
...     load_cc_general,
...     load_online_retail,
... )
>>> df_class = load_online_shoppers()
>>> df_clust = load_cc_general()
>>> df_assoc = load_online_retail()
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Resolve project root regardless of where this module is imported from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UCI URLs (stable, peer-reviewed)
URL_ONLINE_SHOPPERS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00468/online_shoppers_intention.csv"
)
URL_ONLINE_RETAIL_II = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00502/online_retail_II.xlsx"
)

# Local cache filenames
F_ONLINE_SHOPPERS = DATA_DIR / "online_shoppers_intention.csv"
F_CC_GENERAL = DATA_DIR / "CC_GENERAL.csv"
F_ONLINE_RETAIL = DATA_DIR / "online_retail_II.xlsx"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Stream a remote file to disk with a simple progress message."""
    print(f"Downloading {url} -> {dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            written = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
            if total:
                print(f"  ... wrote {written:,} bytes (expected {total:,})")
            else:
                print(f"  ... wrote {written:,} bytes")
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _running_in_colab() -> bool:
    """Detect whether we're inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------

def load_online_shoppers(force_redownload: bool = False) -> pd.DataFrame:
    """
    Load the Online Shoppers Purchasing Intention dataset.

    Source : UCI Machine Learning Repository (id 468)
    Reference: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y.
        (2019). Real-time prediction of online shoppers' purchasing
        intention using multilayer perceptron and LSTM recurrent neural
        networks. Neural Computing and Applications, 31(10), 6893-6908.

    Target column: 'Revenue' (bool) -- whether the session ended in a
    purchase. Class imbalance: ~15% positive.

    Returns
    -------
    pd.DataFrame
        Shape (12330, 18). 17 features + 1 target.
    """
    if force_redownload or not F_ONLINE_SHOPPERS.exists():
        _download_file(URL_ONLINE_SHOPPERS, F_ONLINE_SHOPPERS)
    df = pd.read_csv(F_ONLINE_SHOPPERS)
    print(f"Loaded Online Shoppers: shape={df.shape}, target balance:")
    print(df["Revenue"].value_counts(normalize=True).round(4).to_string())
    return df


def load_cc_general(local_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the CC General (Credit Card Customers) dataset for clustering.

    Source : Kaggle ("Credit Card Dataset for Clustering" by Arjun Bhasin)
    Note   : Kaggle requires authentication for direct download. This
             loader tries three strategies in order:
               1. Use the cached file in DATA_DIR if present.
               2. Use `local_path` if provided.
               3. In Colab, prompt for manual file upload.
               4. Otherwise, raise with clear instructions.

    The dataset contains 8950 customers and 18 features (no target):
    BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, etc.

    Returns
    -------
    pd.DataFrame
        Shape (8950, 18).
    """
    # 1. Cached file
    if F_CC_GENERAL.exists():
        df = pd.read_csv(F_CC_GENERAL)
        print(f"Loaded CC General from cache: shape={df.shape}")
        return df

    # 2. Explicit local path
    if local_path is not None and Path(local_path).exists():
        df = pd.read_csv(local_path)
        df.to_csv(F_CC_GENERAL, index=False)  # cache it
        print(f"Loaded CC General from {local_path}: shape={df.shape}")
        return df

    # 3. Colab interactive upload
    if _running_in_colab():
        print(
            "\nCC General dataset not found in cache.\n"
            "Please upload 'CC GENERAL.csv' (downloadable from Kaggle:\n"
            "  https://www.kaggle.com/datasets/arjunbhasin2013/ccdata\n"
            ") via the file picker below.\n"
        )
        from google.colab import files  # type: ignore
        uploaded = files.upload()
        # Take the first uploaded CSV
        for fname in uploaded:
            if fname.lower().endswith(".csv"):
                df = pd.read_csv(fname)
                df.to_csv(F_CC_GENERAL, index=False)
                print(f"Cached as {F_CC_GENERAL.name}: shape={df.shape}")
                return df
        raise FileNotFoundError("No CSV file was uploaded.")

    # 4. Local non-Colab fallback
    raise FileNotFoundError(
        "CC General dataset not found.\n\n"
        "To obtain it:\n"
        "  1. Download 'CC GENERAL.csv' from\n"
        "     https://www.kaggle.com/datasets/arjunbhasin2013/ccdata\n"
        f"  2. Place it at: {F_CC_GENERAL}\n"
        "  3. Re-run this loader.\n"
    )


def load_online_retail(
    force_redownload: bool = False,
    sheet: str = "Year 2010-2011",
) -> pd.DataFrame:
    """
    Load the Online Retail II dataset for association rule mining.

    Source : UCI Machine Learning Repository (id 502)
    Reference: Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for
        the online retail industry: A case study of RFM model-based
        customer segmentation using data mining. Journal of Database
        Marketing & Customer Strategy Management, 19(3), 197-208.

    The original file contains two sheets ('Year 2009-2010' and
    'Year 2010-2011'). We default to the 2010-2011 sheet (~540K
    transactions) which matches most published benchmarks.

    Returns
    -------
    pd.DataFrame
        Raw transactional records with columns:
        Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
        Customer ID, Country.
    """
    if force_redownload or not F_ONLINE_RETAIL.exists():
        _download_file(URL_ONLINE_RETAIL_II, F_ONLINE_RETAIL)
    print(f"Reading {F_ONLINE_RETAIL.name} (sheet='{sheet}') ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(F_ONLINE_RETAIL, sheet_name=sheet, engine="openpyxl")
    print(f"Loaded Online Retail II: shape={df.shape}")
    return df


# ----------------------------------------------------------------------
# Convenience: load all three with simple diagnostics
# ----------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    """
    Load all three datasets. Returns a dict keyed by task name.
    Useful as a quick smoke test from a notebook.
    """
    out: dict[str, pd.DataFrame] = {}
    print("=" * 60)
    print("LOADING DATASET 1 / 3 : Online Shoppers (classification)")
    print("=" * 60)
    out["classification"] = load_online_shoppers()

    print("\n" + "=" * 60)
    print("LOADING DATASET 2 / 3 : CC General (clustering)")
    print("=" * 60)
    out["clustering"] = load_cc_general()

    print("\n" + "=" * 60)
    print("LOADING DATASET 3 / 3 : Online Retail II (association)")
    print("=" * 60)
    out["association"] = load_online_retail()

    print("\nAll datasets loaded successfully.")
    return out


if __name__ == "__main__":
    # Smoke test: run `python -m src.data_loader` from the project root.
    try:
        datasets = load_all()
        for name, df in datasets.items():
            print(f"\n--- {name} preview ---")
            print(df.head(3))
            print(f"dtypes:\n{df.dtypes}")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
