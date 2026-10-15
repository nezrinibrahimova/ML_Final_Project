from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decision_tree import DecisionTree
from src import evaluate

SEED = 42
FIG_DIR = "report/figures"
DEPTH_GRID = [6, 8, 10, 12, 15]
RARE_TRAIN_COUNT = 30        # district with fewer train listings = "rare"


def load_splits(synthetic: bool):
    if synthetic:
        rng = np.random.default_rng(SEED)
        n = 6000
        X = rng.normal(size=(n, 8))
        price = np.exp(11 + 0.8 * X[:, 0] + 0.4 * np.sin(2 * X[:, 1]) + 0.3 * X[:, 2] * X[:, 3] + rng.normal(0, .3, n))
        idx = rng.permutation(n); a, b = int(.7 * n), int(.85 * n)
        tr, va, te = idx[:a], idx[a:b], idx[b:]
        return X[tr], price[tr], X[va], price[va], X[te], price[te], [f"f{i}" for i in range(8)]
    from src import data_prep
    df = data_prep.clean(data_prep.load_raw())
    X, y, names = data_prep.make_features(df)
    quart = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    return (*data_prep.train_val_test_split(X, y, stratify=quart, seed=SEED), names)


def decode_onehot(X, names, prefix):
    """Recover a categorical column from its one-hot block (None if absent)."""
    cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
    if not cols:
        return None
    return np.array([names[cols[k]][len(prefix):] for k in X[:, cols].argmax(axis=1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    X_tr, p_tr, X_va, p_va, X_te, p_te, names = load_splits(args.synthetic)
    l_tr, l_va, l_te = np.log(p_tr), np.log(p_va), np.log(p_te)

    #Task A 
    best_d = min(DEPTH_GRID, key=lambda d: evaluate.rmse(
        l_va, DecisionTree("regression", "mse", max_depth=d, random_state=SEED).fit(X_tr, l_tr).predict(X_va)))
    model = DecisionTree("regression", "mse", max_depth=best_d, random_state=SEED).fit(X_tr, l_tr)
    pred_log = model.predict(X_te)
    pred = np.exp(pred_log)
    print(f"Task A: max_depth={best_d} (picked on validation) | test RMSE(log)={evaluate.rmse(l_te, pred_log):.4f}"
          f" | test MAE(AZN)={evaluate.mae(p_te, pred):,.0f}")

    err = pred - p_te
    df = pd.DataFrame({"price": p_te, "pred": pred, "abs_err": np.abs(err),
                       "pct_err": np.abs(err) / p_te * 100, "log_err": pred_log - l_te})
    loc_te = decode_onehot(X_te, names, "location_")
    cat_te = decode_onehot(X_te, names, "Kateqoriya_")
    if loc_te is not None: df["district"] = loc_te
    if cat_te is not None: df["category"] = cat_te
    for key in ("area_m2", "rooms", "floor"):
        if key in names: df[key] = X_te[:, names.index(key)]

    # 1) by price segment
    seg_edges = np.quantile(p_te, [0, .5, .9, .99, 1.0])
    df["segment"] = pd.cut(df["price"], seg_edges, include_lowest=True,
                           labels=["bottom 50%", "50-90%", "90-99%", "top 1% (luxury)"])
    seg = df.groupby("segment", observed=True).agg(
        n=("price", "size"), mean_price=("price", "mean"), MAE_AZN=("abs_err", "mean"),
        median_pct_err=("pct_err", "median"), bias_log=("log_err", "mean")).round(3)
    print("\n1) Error by true-price segment (bias_log < 0 -> tree UNDER-predicts):")
    print(seg.to_string())
    seg.to_csv(os.path.join(FIG_DIR, "tree_error_by_segment.csv"))

    # 2) by district
    if "district" in df:
        train_loc = decode_onehot(X_tr, names, "location_")
        counts = pd.Series(train_loc).value_counts()
        df["train_count"] = df["district"].map(counts).fillna(0).astype(int)
        d = df.groupby("district").agg(n_test=("price", "size"), train_n=("train_count", "first"),
                                       median_pct_err=("pct_err", "median"), bias_log=("log_err", "mean")).round(3)
        d = d[d["n_test"] >= 5].sort_values("median_pct_err", ascending=False)
        print(f"\n2) Districts with >=5 test rows, worst 10 by median % error:")
        print(d.head(10).to_string())
        rare = df[df["train_count"] < RARE_TRAIN_COUNT]; common = df[df["train_count"] >= RARE_TRAIN_COUNT]
        if len(rare) and len(common):
            print(f"\n   rare districts (<{RARE_TRAIN_COUNT} train rows): n={len(rare)} median % err={rare['pct_err'].median():.1f}"
                  f" | common: n={len(common)} median % err={common['pct_err'].median():.1f}")
        d.to_csv(os.path.join(FIG_DIR, "tree_error_by_district.csv"))

    # 3) worst predictions
    worst = df.sort_values("abs_err", ascending=False).head(15)
    cols = [c for c in ["price", "pred", "pct_err", "area_m2", "rooms", "floor", "district", "category"] if c in worst]
    print("\n3) 15 worst test predictions (largest absolute AZN error):")
    print(worst[cols].round(1).to_string(index=False))
    worst[cols].round(1).to_csv(os.path.join(FIG_DIR, "tree_error_worst15.csv"), index=False)

    #Task B
    thr = float(np.median(p_tr))
    c_tr, c_va, c_te = ((p > thr).astype(int) for p in (p_tr, p_va, p_te))
    best = max(DEPTH_GRID, key=lambda d: evaluate.precision_recall_f1(
        c_va, DecisionTree("classification", "gini", max_depth=d, random_state=SEED).fit(X_tr, c_tr).predict(X_va))[2])
    clf = DecisionTree("classification", "gini", max_depth=best, random_state=SEED).fit(X_tr, c_tr)
    wrong = clf.predict(X_te) != c_te
    dist = np.abs(np.log(p_te / thr))
    bins = pd.cut(dist, [0, .1, .25, .5, 1.0, np.inf], include_lowest=True,
                  labels=["<10%", "10-25%", "25-50%", "50-100%", ">100%"])
    b = pd.DataFrame({"bin": bins, "wrong": wrong}).groupby("bin", observed=True).agg(
        n=("wrong", "size"), error_rate=("wrong", "mean")).round(3)
    print(f"\n4) Task B (gini, depth {best}): error rate by distance from tier threshold {thr:,.0f} AZN"
          f" (|log(price/threshold)|):")
    print(b.to_string())
    print(f"   overall test error rate = {wrong.mean():.3f}")
    b.to_csv(os.path.join(FIG_DIR, "tree_error_B_by_distance.csv"))

    #figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(seg.index.astype(str), seg["median_pct_err"]); ax[0].set_title("Task A: median % error by price segment")
    ax[0].set_ylabel("median |error| / price (%)"); ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(b.index.astype(str), b["error_rate"], color="tab:orange"); ax[1].set_title("Task B: error rate vs distance from threshold")
    ax[1].set_xlabel("|log(price / threshold)|"); ax[1].set_ylabel("error rate")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "tree_error_analysis.png"), dpi=200)
    print(f"\nSaved tree_error_*.csv and tree_error_analysis.png in {FIG_DIR}/")


if __name__ == "__main__":
    main()