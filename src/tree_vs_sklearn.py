from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import evaluate
from src.decision_tree import DecisionTree
from src.tree_common import SEED, TABLES, load_splits, select_config, utf8_console


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def top_features(imp, names, k=8):
    order = np.argsort(imp)[::-1][:k]
    return [(names[i], float(imp[i])) for i in order]


def print_importances(title, mine, sk, names):
    a, b = top_features(mine, names), top_features(sk, names)
    print(f"\n{title}: top-8 feature importances (mine | sklearn)")
    for (n1, v1), (n2, v2) in zip(a, b):
        print(f"  {n1[:28]:<28} {v1:.3f}   |   {n2[:28]:<28} {v2:.3f}")
    overlap = len({n for n, _ in top_features(mine, names, 10)} & {n for n, _ in top_features(sk, names, 10)})
    print(f"  top-10 overlap: {overlap}/10")
    return overlap


def reg_row(p_te, l_te, pred_log, fit_s, pred_s):
    pred = np.exp(pred_log)
    return {"RMSE": evaluate.rmse(p_te, pred), "MAE": evaluate.mae(p_te, pred), "R2": evaluate.r2(p_te, pred),
            "RMSE(log)": evaluate.rmse(l_te, pred_log), "R2(log)": evaluate.r2(l_te, pred_log),
            "fit s": fit_s, "predict s": pred_s}


def cls_row(c_te, pred, prob, fit_s, pred_s):
    p, r, f1 = evaluate.precision_recall_f1(c_te, pred)
    return {"Acc": float(np.mean(pred == c_te)), "Prec": p, "Rec": r, "F1": f1,
            "AUC": evaluate.roc_auc(c_te, prob), "fit s": fit_s, "predict s": pred_s}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv if argv is not None else [])
    utf8_console()
    TABLES.mkdir(parents=True, exist_ok=True)

    d = load_splits(args.synthetic)
    X_tr, X_te, names = d["X_tr"], d["X_te"], d["names"]
    l_tr, l_te = np.log(d["p_tr"]), np.log(d["p_te"])
    print(f"train={len(X_tr)} val={len(d['X_va'])} test={len(X_te)} features={X_tr.shape[1]} "
          f"tier threshold={d['threshold']:,.0f}")
    cfg = select_config(d)
    detail = []

    # ---------------- Task A ----------------
    cA = cfg["A"]
    mine, fit_m = timed(lambda: DecisionTree("regression", "mse", random_state=SEED, **cA).fit(X_tr, l_tr))
    sk, fit_s = timed(lambda: DecisionTreeRegressor(random_state=SEED, **cA).fit(X_tr, l_tr))
    pm, pt_m = timed(lambda: mine.predict(X_te))
    ps, pt_s = timed(lambda: sk.predict(X_te))
    tab_a = pd.DataFrame({"Decision tree (ours)": reg_row(d["p_te"], l_te, pm, fit_m, pt_m),
                          "sklearn DecisionTreeRegressor": reg_row(d["p_te"], l_te, ps, fit_s, pt_s)}).T
    tab_a.to_csv(TABLES / "tree_task_a_results.csv")
    print(f"\n=== Task A ({cA}, chosen on validation) — TEST ===")
    evaluate.compare({k: v.to_dict() for k, v in tab_a.iterrows()})
    print(f"  size: mine depth={mine.get_depth()} leaves={mine.get_n_leaves()} | "
          f"sklearn depth={sk.get_depth()} leaves={sk.get_n_leaves()}")
    print(f"  prediction agreement: corr={np.corrcoef(pm, ps)[0, 1]:.4f}, mean|diff|={np.mean(np.abs(pm - ps)):.4f}")
    ov = print_importances("Task A", mine.feature_importances_, sk.feature_importances_, names)
    detail.append({"task": "A", "criterion": "mse", "max_depth": cA["max_depth"], "min_samples_leaf": cA["min_samples_leaf"], "leaves_mine": mine.get_n_leaves(),
                   "leaves_sklearn": sk.get_n_leaves(), "agreement_or_corr": float(np.corrcoef(pm, ps)[0, 1]),
                   "top10_importance_overlap": ov})

    # ---------------- Task B ----------------
    rows = {}
    for crit in ("gini", "entropy"):
        cB = cfg[f"B-{crit}"]
        mine, fit_m = timed(lambda: DecisionTree("classification", crit, random_state=SEED, **cB).fit(X_tr, d["c_tr"]))
        sk, fit_s = timed(lambda: DecisionTreeClassifier(criterion=crit, random_state=SEED, **cB).fit(X_tr, d["c_tr"]))
        pm, pt_m = timed(lambda: mine.predict(X_te))
        ps, pt_s = timed(lambda: sk.predict(X_te))
        rows[f"Decision tree {crit} (ours)"] = cls_row(d["c_te"], pm, mine.predict_proba(X_te)[:, 1], fit_m, pt_m)
        rows[f"sklearn DecisionTree {crit}"] = cls_row(d["c_te"], ps, sk.predict_proba(X_te)[:, 1], fit_s, pt_s)
        print(f"\n=== Task B, {crit}, {cB} (chosen on validation) — TEST ===")
        evaluate.compare({k: v for k, v in rows.items() if crit in k})
        print("  confusion mine   [[TN FP],[FN TP]]:", evaluate.confusion_matrix(d["c_te"], pm).tolist())
        print("  confusion sklearn[[TN FP],[FN TP]]:", evaluate.confusion_matrix(d["c_te"], ps).tolist())
        print(f"  size: mine depth={mine.get_depth()} leaves={mine.get_n_leaves()} | "
              f"sklearn depth={sk.get_depth()} leaves={sk.get_n_leaves()}")
        agree = float(np.mean(pm == ps))
        print(f"  prediction agreement: {agree:.4f}")
        ov = print_importances(f"Task B ({crit})", mine.feature_importances_, sk.feature_importances_, names)
        detail.append({"task": "B", "criterion": crit, "max_depth": cB["max_depth"], "min_samples_leaf": cB["min_samples_leaf"], "leaves_mine": mine.get_n_leaves(),
                       "leaves_sklearn": sk.get_n_leaves(), "agreement_or_corr": agree,
                       "top10_importance_overlap": ov})
    pd.DataFrame(rows).T.to_csv(TABLES / "tree_task_b_results.csv")
    pd.DataFrame(detail).to_csv(TABLES / "tree_vs_sklearn.csv", index=False)
    print(f"\nSaved tree_task_a_results.csv, tree_task_b_results.csv, tree_vs_sklearn.csv in {TABLES}/")


if __name__ == "__main__":
    main(sys.argv[1:])
