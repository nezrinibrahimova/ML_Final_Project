"""data_prep.py: loading, cleaning, features, split, tier label, standardize."""
from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 42
DATA_PATH = "data/house_sale.csv"
LEAKAGE_COLUMNS: list[str] = ["total_price", "unit_price"]
KEEP_CATEGORIES = ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi"]
NUMERIC_FEATURES = ["area_m2", "rooms", "floor", "floors_total", "land_sot",
                    "lat", "lng", "has_repair", "has_deed", "has_mortgage"]


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame, unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def clean(df):
    """Residential listings only, latest record per listing, numeric columns,
    outlier rules, missing values, leakage columns dropped."""
    df = df.copy()
    df = df[df["Kateqoriya"].isin(KEEP_CATEGORIES)]
    df["_t"] = pd.to_datetime(df["datetime_scrape_x"], format="mixed", utc=True)
    df = df.sort_values("_t").drop_duplicates("estate_rel_url_x", keep="last")
    df = df.drop(columns="_t")
    df["area_m2"] = pd.to_numeric(df["Sahə"].str.extract(r"([\d.]+)")[0], errors="coerce")
    fl = df["Mərtəbə"].str.extract(r"(\d+)\s*/\s*(\d+)")
    df["floor"] = pd.to_numeric(fl[0], errors="coerce")
    df["floors_total"] = pd.to_numeric(fl[1], errors="coerce")
    df["land_sot"] = pd.to_numeric(df["Torpaq sahəsi"].str.extract(r"([\d.]+)")[0], errors="coerce")
    df["ppm2"] = df["price"] / df["area_m2"]
    df = df[df["area_m2"].between(15, 1000)]
    df = df[df["ppm2"].between(300, 10000)]
    df = df[~(df["land_sot"] > 1000)]
    df = df.drop(columns="ppm2")
    df = df[df["Otaq sayı"].notna()]
    df[["floor", "floors_total", "land_sot"]] = df[["floor", "floors_total", "land_sot"]].fillna(0)
    df["rooms"] = df["Otaq sayı"]
    df["has_repair"] = (df["Təmir"] == "var").astype(float)
    df["has_deed"] = (df["Çıxarış"] == "var").astype(float)
    df["has_mortgage"] = (df["İpoteka"] == "var").astype(float)
    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    df = df.dropna(subset=["lat", "lng", "location"])
    # Remove clearly invalid geographic coordinates
    df = df[
        df["lat"].between(38.0, 42.0)
        & df["lng"].between(44.0, 51.0)
    ]
    keep = ["price", "Kateqoriya", "location"] + NUMERIC_FEATURES
    return df[keep].reset_index(drop=True)


def make_features(df):
    """Return (X, y, feature_names). One-hot for category and district."""
    y = df["price"].to_numpy(dtype=float)
    cats = pd.get_dummies(df[["Kateqoriya", "location"]], dtype=float)
    X_df = pd.concat([df[NUMERIC_FEATURES].astype(float), cats], axis=1)
    return X_df.to_numpy(dtype=float), y, list(X_df.columns)


def make_tier_label(y_price, threshold=None):
    """premium=1 if price > threshold. Threshold = TRAIN median if not given."""
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED, stratify=None):
    """Deterministic split. Optional `stratify` = group labels (e.g. price quartiles)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if stratify is None:
        groups = [np.arange(n)]
    else:
        s = np.asarray(stratify)
        groups = [np.where(s == g)[0] for g in np.unique(s)]
    tr, va, te = [], [], []
    for idx in groups:
        idx = rng.permutation(idx)
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    tr, va, te = (rng.permutation(np.concatenate(a)) for a in (tr, va, te))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def standardize(X_tr, *others):
    """Standardize using TRAIN mean/std only."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return tuple((a - mu) / sd for a in (X_tr, *others))