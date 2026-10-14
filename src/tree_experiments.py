from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decision_tree import DecisionTree
from src import evaluate

SEED = 42
FIG_DIR = "report/figures"
DEPTHS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
LEAFS = [1, 5, 10, 25, 50, 100, 200]


def _try(fn, fallback, *a):
    try:
        return fn(*a)
    except NotImplementedError:
        return fallback(*a)


def _rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def _f1(y, p):
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def rmse(y, p): return float(_try(evaluate.rmse, _rmse, y, p))


def f1(y, p):
    try:
        return float(evaluate.precision_recall_f1(y, p)[2])
    except NotImplementedError:
        return float(_f1(y, p))



def get_data(synthetic: bool):
    """Return dict with X_tr, X_val, y_tr_reg, y_val_reg, y_tr_cls, y_val_cls, names.
    Regression target is log(price). Tier threshold comes from TRAIN only."""
    if synthetic:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(6000, 8))
        logp = 11 + 0.8 * X[:, 0] + 0.4 * np.sin(2 * X[:, 1]) + 0.3 * X[:, 2] * X[:, 3] + rng.normal(0, .3, 6000)
        names = [f"f{i}" for i in range(8)]
        idx = rng.permutation(6000); tr, va = idx[:4200], idx[4200:5100]
        X_tr, X_val, lp_tr, lp_val = X[tr], X[va], logp[tr], logp[va]
        thr = np.median(np.exp(lp_tr))
        return dict(X_tr=X_tr, X_val=X_val, y_tr_reg=lp_tr, y_val_reg=lp_val,
                    y_tr_cls=(np.exp(lp_tr) > thr).astype(int),
                    y_val_cls=(np.exp(lp_val) > thr).astype(int), names=names)

    from src import data_prep
    df = data_prep.clean(data_prep.load_raw())       
    X, y, names = data_prep.make_features(df)
    
    quart = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    X_tr, y_tr, X_val, y_val, _X_te, _y_te = data_prep.train_val_test_split(
        X, y, stratify=quart, seed=SEED)
    y_tr_cls, thr = data_prep.make_tier_label(y_tr)             
    y_val_cls, _ = data_prep.make_tier_label(y_val, thr)        
    print(f"tier threshold (train median price) = {thr:,.0f}")
    return dict(X_tr=X_tr, X_val=X_val,
                y_tr_reg=np.log(y_tr), y_val_reg=np.log(y_val),
                y_tr_cls=y_tr_cls, y_val_cls=y_val_cls, names=names)



def sweep(make_model, param_name, values, d, task):
    """Fit one model per value; return list of (value, train_score, val_score, n_leaves)."""
    rows = []
    for v in values:
        m = make_model(v).fit(d["X_tr"], d[f"y_tr_{task}"])
        ptr, pva = m.predict(d["X_tr"]), m.predict(d["X_val"])
        sc = rmse if task == "reg" else f1
        rows.append((v, sc(d[f"y_tr_{task}"], ptr), sc(d[f"y_val_{task}"], pva), m.get_n_leaves()))
        print(f"  {param_name}={v!s:>5}  train={rows[-1][1]:.4f}  val={rows[-1][2]:.4f}  leaves={rows[-1][3]}")
    return rows


def plot_curves(series, xlabel, ylabel, title, fname, logx=False):
    plt.figure(figsize=(6, 4))
    for label, rows, style in series:
        x = [r[0] for r in rows]
        plt.plot(x, [r[1] for r in rows], style[0], label=f"{label} train", alpha=.6)
        plt.plot(x, [r[2] for r in rows], style[1], label=f"{label} val")
    if logx: plt.xscale("log")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=200); plt.close()


def learning_curve(make_model, d, task, fracs=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0)):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(d["X_tr"]))
    sc = rmse if task == "reg" else f1
    rows = []
    for f in fracs:
        idx = order[: max(int(f * len(order)), 20)]
        m = make_model().fit(d["X_tr"][idx], d[f"y_tr_{task}"][idx])
        rows.append((len(idx), sc(d[f"y_tr_{task}"][idx], m.predict(d["X_tr"][idx])),
                     sc(d[f"y_val_{task}"], m.predict(d["X_val"])), m.get_n_leaves()))
        print(f"  n={len(idx):>7}  train={rows[-1][1]:.4f}  val={rows[-1][2]:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    d = get_data(args.synthetic)
    print(f"train={len(d['X_tr'])} val={len(d['X_val'])} features={d['X_tr'].shape[1]}")
    out = []

    #Task B: gini vs entropy over depth
    print("\nTask B: criterion x max_depth (F1)")
    series = []
    best_b = None
    for crit, st in (("gini", ("b--", "b-")), ("entropy", ("r--", "r-"))):
        print(f" criterion={crit}")
        rows = sweep(lambda v: DecisionTree("classification", crit, max_depth=v, random_state=SEED),
                     "max_depth", DEPTHS, d, "cls")
        series.append((crit, rows, st))
        for r in rows:
            out.append(("B", crit, "max_depth", *r))
            if best_b is None or r[2] > best_b[0]:
                best_b = (r[2], crit, r[0])
    plot_curves(series, "max_depth", "F1 (premium class)", "Task B: Gini vs entropy", "tree_B_depth.png")

    #Task B: extra hyperparameter, min_samples_leaf
    print(f"\nTask B: min_samples_leaf (criterion={best_b[1]})")
    rows = sweep(lambda v: DecisionTree("classification", best_b[1], min_samples_leaf=v, random_state=SEED),
                 "min_samples_leaf", LEAFS, d, "cls")
    out += [("B", best_b[1], "min_samples_leaf", *r) for r in rows]
    plot_curves([(best_b[1], rows, ("g--", "g-"))], "min_samples_leaf (log)", "F1",
                "Task B: min_samples_leaf", "tree_B_leaf.png", logx=True)

    #Task A: depth and leaf sweeps
    print("\nTask A (log price): max_depth (RMSE)")
    rows_a = sweep(lambda v: DecisionTree("regression", "mse", max_depth=v, random_state=SEED),
                   "max_depth", DEPTHS, d, "reg")
    out += [("A", "mse", "max_depth", *r) for r in rows_a]
    plot_curves([("mse", rows_a, ("b--", "b-"))], "max_depth", "RMSE (log price)",
                "Task A: max_depth", "tree_A_depth.png")
    print("\nTask A: min_samples_leaf (RMSE)")
    rows_l = sweep(lambda v: DecisionTree("regression", "mse", min_samples_leaf=v, random_state=SEED),
                   "min_samples_leaf", LEAFS, d, "reg")
    out += [("A", "mse", "min_samples_leaf", *r) for r in rows_l]
    plot_curves([("mse", rows_l, ("g--", "g-"))], "min_samples_leaf (log)", "RMSE (log price)",
                "Task A: min_samples_leaf", "tree_A_leaf.png", logx=True)

    #Learning curves (deep tree -> variance; shows gap closing with data)
    print("\nLearning curve Task B (depth 10)")
    lc_b = learning_curve(lambda: DecisionTree("classification", best_b[1], max_depth=10, random_state=SEED), d, "cls")
    plot_curves([("depth10", lc_b, ("b--", "b-"))], "training size", "F1", "Task B learning curve", "tree_B_learning.png", logx=True)
    print("\nLearning curve Task A (depth 10)")
    lc_a = learning_curve(lambda: DecisionTree("regression", "mse", max_depth=10, random_state=SEED), d, "reg")
    plot_curves([("depth10", lc_a, ("b--", "b-"))], "training size", "RMSE (log price)", "Task A learning curve", "tree_A_learning.png", logx=True)

    with open(os.path.join(FIG_DIR, "tree_sweeps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task", "criterion", "param", "value", "train", "val", "n_leaves"]); w.writerows(out)

    #Final configs: VALIDATION only
    best_a = min(rows_a, key=lambda r: r[2])
    print("\n=== Picked from validation (no test set used) ===")
    print(f"Task B: criterion={best_b[1]}, max_depth={best_b[2]}  (val F1={best_b[0]:.4f})")
    print(f"Task A: max_depth={best_a[0]}  (val RMSE log-price={best_a[2]:.4f})")
    print(f"Figures + tree_sweeps.csv saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()