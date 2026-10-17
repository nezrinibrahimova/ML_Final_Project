from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import evaluate
from src.decision_tree import DecisionTree
from src.tree_common import FIGURES, SEED, TABLES, load_splits, select_config, utf8_console

RARE_TRAIN_COUNT = 30        # district with fewer train listings = "rare"


def decode_onehot(X, names, prefix):
    """Recover a categorical column from its one-hot block (None if absent)."""
    cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
    if not cols:
        return None
    return np.array([names[cols[k]][len(prefix):] for k in X[:, cols].argmax(axis=1)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv if argv is not None else [])
    utf8_console()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    d = load_splits(args.synthetic)
    X_tr, X_te, names = d["X_tr"], d["X_te"], d["names"]
    p_te, l_tr, l_te = d["p_te"], np.log(d["p_tr"]), np.log(d["p_te"])
    cfg = select_config(d, verbose=False)

    # ---------------- Task A ----------------
    cA = cfg["A"]
    model = DecisionTree("regression", "mse", random_state=SEED, **cA).fit(X_tr, l_tr)
    pred_log = model.predict(X_te)
    pred = np.exp(pred_log)
    print(f"Task A: {cA} (picked on validation) | test RMSE(log)={evaluate.rmse(l_te, pred_log):.4f}"
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
    seg.to_csv(TABLES / "tree_error_by_segment.csv")

    # 2) by district
    if "district" in df:
        train_loc = decode_onehot(X_tr, names, "location_")
        counts = pd.Series(train_loc).value_counts()
        df["train_count"] = df["district"].map(counts).fillna(0).astype(int)
        dist = df.groupby("district").agg(n_test=("price", "size"), train_n=("train_count", "first"),
                                          median_pct_err=("pct_err", "median"), bias_log=("log_err", "mean")).round(3)
        dist = dist[dist["n_test"] >= 5].sort_values("median_pct_err", ascending=False)
        print("\n2) Districts with >=5 test rows, worst 10 by median % error:")
        print(dist.head(10).to_string())
        rare = df[df["train_count"] < RARE_TRAIN_COUNT]; common = df[df["train_count"] >= RARE_TRAIN_COUNT]
        if len(rare) and len(common):
            print(f"\n   rare districts (<{RARE_TRAIN_COUNT} train rows): n={len(rare)} median % err={rare['pct_err'].median():.1f}"
                  f" | common: n={len(common)} median % err={common['pct_err'].median():.1f}")
        dist.to_csv(TABLES / "tree_error_by_district.csv")

    # 3) worst predictions
    worst = df.sort_values("abs_err", ascending=False).head(15)
    cols = [c for c in ["price", "pred", "pct_err", "area_m2", "rooms", "floor", "district", "category"] if c in worst]
    print("\n3) 15 worst test predictions (largest absolute AZN error):")
    print(worst[cols].round(1).to_string(index=False))
    worst[cols].round(1).to_csv(TABLES / "tree_error_worst15.csv", index=False)

    # ---------------- Task B ----------------
    thr, c_te = d["threshold"], d["c_te"]
    cB = cfg["B-gini"]
    clf = DecisionTree("classification", "gini", random_state=SEED, **cB).fit(X_tr, d["c_tr"])
    wrong = clf.predict(X_te) != c_te
    dist_thr = np.abs(np.log(p_te / thr))
    bins = pd.cut(dist_thr, [0, .1, .25, .5, 1.0, np.inf], include_lowest=True,
                  labels=["<10%", "10-25%", "25-50%", "50-100%", ">100%"])
    b = pd.DataFrame({"bin": bins, "wrong": wrong}).groupby("bin", observed=True).agg(
        n=("wrong", "size"), error_rate=("wrong", "mean")).round(3)
    print(f"\n4) Task B (gini, {cB}): error rate by distance from tier threshold {thr:,.0f} AZN"
          f" (|log(price/threshold)|):")
    print(b.to_string())
    print(f"   overall test error rate = {wrong.mean():.3f}")
    b.to_csv(TABLES / "tree_error_B_by_distance.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(seg.index.astype(str), seg["median_pct_err"]); ax[0].set_title("Task A: median % error by price segment")
    ax[0].set_ylabel("median |error| / price (%)"); ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(b.index.astype(str), b["error_rate"], color="tab:orange"); ax[1].set_title("Task B: error rate vs distance from threshold")
    ax[1].set_xlabel("|log(price / threshold)|"); ax[1].set_ylabel("error rate")
    plt.tight_layout(); plt.savefig(FIGURES / "tree_error_analysis.png", dpi=200); plt.close()
    print(f"\nSaved tree_error_*.csv in {TABLES}/ and tree_error_analysis.png in {FIGURES}/")


if __name__ == "__main__":
    main(sys.argv[1:])
