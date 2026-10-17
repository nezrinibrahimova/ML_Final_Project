from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import evaluate
from src.decision_tree import DecisionTree
from src.tree_common import SEED, TABLES, load_splits, select_config, utf8_console


def summarize(records: dict) -> pd.DataFrame:
    """records: {model: {metric: [values per fold]}} -> table with mean and std columns."""
    out = {}
    for model, metrics in records.items():
        row = {}
        for k, v in metrics.items():
            row[f"{k} mean"] = float(np.mean(v))
            row[f"{k} std"] = float(np.std(v, ddof=1))
        out[model] = row
    return pd.DataFrame(out).T


def print_pm(title, table, metrics):
    print(f"\n{title}")
    width = max(len(m) for m in table.index)
    for model, row in table.iterrows():
        cells = "  ".join(f"{k} {row[f'{k} mean']:.4f} ± {row[f'{k} std']:.4f}" for k in metrics)
        print(f"  {model:<{width}}  {cells}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv if argv is not None else [])
    utf8_console()
    TABLES.mkdir(parents=True, exist_ok=True)

    d = load_splits(args.synthetic)
    X, p, c = d["X_tr"], d["p_tr"], d["c_tr"]
    l = np.log(p)
    cfg = select_config(d, verbose=False)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    groups = pd.qcut(p, 4, labels=False)           # Task A folds balanced by price quartile
    print(f"{args.folds}-fold CV on the train split (n={len(X)}); config {cfg}")

    # ---------------- Task A ----------------
    rec = {"Decision tree (ours)": {"RMSE(log)": [], "R2(log)": []},
           "sklearn DecisionTreeRegressor": {"RMSE(log)": [], "R2(log)": []}}
    for tr, va in skf.split(X, groups):
        for name, model in (("Decision tree (ours)", DecisionTree("regression", "mse", random_state=SEED, **cfg["A"])),
                            ("sklearn DecisionTreeRegressor", DecisionTreeRegressor(random_state=SEED, **cfg["A"]))):
            pred = model.fit(X[tr], l[tr]).predict(X[va])
            rec[name]["RMSE(log)"].append(evaluate.rmse(l[va], pred))
            rec[name]["R2(log)"].append(evaluate.r2(l[va], pred))
    tab_a = summarize(rec)
    print_pm(f"Task A ({cfg['A']}) — CV mean ± std", tab_a, ["RMSE(log)", "R2(log)"])

    # ---------------- Task B ----------------
    rec_b, tab_parts = {}, [tab_a]
    for crit in ("gini", "entropy"):
        cB = cfg[f"B-{crit}"]
        names = {f"Decision tree {crit} (ours)": DecisionTree("classification", crit, random_state=SEED, **cB),
                 f"sklearn DecisionTree {crit}": DecisionTreeClassifier(criterion=crit, random_state=SEED, **cB)}
        for name in names:
            rec_b[name] = {"F1": [], "AUC": [], "Acc": []}
        for tr, va in skf.split(X, c):
            for name, model in names.items():
                model.fit(X[tr], c[tr])
                pred, prob = model.predict(X[va]), model.predict_proba(X[va])[:, 1]
                rec_b[name]["F1"].append(evaluate.precision_recall_f1(c[va], pred)[2])
                rec_b[name]["AUC"].append(evaluate.roc_auc(c[va], prob))
                rec_b[name]["Acc"].append(float(np.mean(pred == c[va])))
    tab_b = summarize(rec_b)
    print_pm("Task B — CV mean ± std", tab_b, ["F1", "AUC", "Acc"])
    pd.concat([tab_a, tab_b]).to_csv(TABLES / "tree_cv.csv")

    # ---------------- scaling ----------------
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X))
    rows = []
    X_te = d["X_te"]
    for frac in (0.1, 0.25, 0.5, 1.0):
        idx = order[: max(int(frac * len(order)), 50)]
        r = {"n_train": len(idx)}
        for name, mk in (("ours", lambda: DecisionTree("regression", "mse", random_state=SEED, **cfg["A"])),
                         ("sklearn", lambda: DecisionTreeRegressor(random_state=SEED, **cfg["A"]))):
            m = mk()
            t = time.perf_counter(); m.fit(X[idx], l[idx]); r[f"{name} fit s"] = time.perf_counter() - t
            t = time.perf_counter(); m.predict(X_te); r[f"{name} predict s"] = time.perf_counter() - t
        r["fit ratio (ours/sklearn)"] = r["ours fit s"] / max(r["sklearn fit s"], 1e-9)
        rows.append(r)
    scaling = pd.DataFrame(rows)
    scaling.to_csv(TABLES / "tree_scaling.csv", index=False)
    print(f"\nFit/predict time vs training size (Task A, {cfg['A']}; predict = full test set):")
    print(scaling.round(3).to_string(index=False))
    print(f"\nSaved tree_cv.csv and tree_scaling.csv in {TABLES}/")


if __name__ == "__main__":
    main(sys.argv[1:])
