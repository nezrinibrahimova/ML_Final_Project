from __future__ import annotations

import argparse
import csv
import os
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree
from src import evaluate

SEED = 42
FIG_DIR = "report/figures"
DEPTH_GRID = [6, 8, 10, 12, 15]


def load_splits(synthetic: bool):
    """Return train/val/test features, price targets and feature names."""
    if synthetic:
        rng = np.random.default_rng(SEED)
        n = 6000
        X = rng.normal(size=(n, 8))
        price = np.exp(11 + 0.8 * X[:, 0] + 0.4 * np.sin(2 * X[:, 1]) + 0.3 * X[:, 2] * X[:, 3]
                       + rng.normal(0, .3, n))
        idx = rng.permutation(n)
        a, b = int(.7 * n), int(.85 * n)
        tr, va, te = idx[:a], idx[a:b], idx[b:]
        return (X[tr], price[tr], X[va], price[va], X[te], price[te], [f"f{i}" for i in range(8)])

    from src import data_prep
    df = data_prep.clean(data_prep.load_raw())
    X, y, names = data_prep.make_features(df)
    quart = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))      # same split as tree_experiments
    X_tr, y_tr, X_va, y_va, X_te, y_te = data_prep.train_val_test_split(X, y, stratify=quart, seed=SEED)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, names


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def tree_size(model, mine: bool):
    if mine:
        return model.get_depth(), model.get_n_leaves()
    return model.get_depth(), model.get_n_leaves()


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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    X_tr, p_tr, X_va, p_va, X_te, p_te, names = load_splits(args.synthetic)
    thr = float(np.median(p_tr))                                    # tier threshold: TRAIN median only
    c_tr, c_va, c_te = ((p > thr).astype(int) for p in (p_tr, p_va, p_te))
    l_tr, l_va, l_te = np.log(p_tr), np.log(p_va), np.log(p_te)     # Task A target = log(price)
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)} features={X_tr.shape[1]} tier thr={thr:,.0f}")

    rows = []

    # ============================ Task A: regression ============================
    print("\n=== Task A (log price): choose max_depth on VALIDATION with my tree ===")
    best_d, best_v = None, np.inf
    for d in DEPTH_GRID:
        m = DecisionTree("regression", "mse", max_depth=d, random_state=SEED).fit(X_tr, l_tr)
        v = evaluate.rmse(l_va, m.predict(X_va))
        print(f"  max_depth={d:>2}  val RMSE(log)={v:.4f}")
        if v < best_v:
            best_d, best_v = d, v
    print(f"  -> chosen max_depth={best_d} (used for BOTH models)")

    mine, t_fit_m = timed(lambda: DecisionTree("regression", "mse", max_depth=best_d, random_state=SEED).fit(X_tr, l_tr))
    sk, t_fit_s = timed(lambda: DecisionTreeRegressor(max_depth=best_d, random_state=SEED).fit(X_tr, l_tr))
    (pm, t_pr_m), (ps, t_pr_s) = timed(lambda: mine.predict(X_te)), timed(lambda: sk.predict(X_te))

    def reg_metrics(pred_log):
        pred_azn = np.exp(pred_log)
        return {"RMSE(log)": evaluate.rmse(l_te, pred_log), "R2(log)": evaluate.r2(l_te, pred_log),
                "RMSE(AZN)": evaluate.rmse(p_te, pred_azn), "MAE(AZN)": evaluate.mae(p_te, pred_azn),
                "R2(AZN)": evaluate.r2(p_te, pred_azn)}

    res = {"my tree": reg_metrics(pm), "sklearn": reg_metrics(ps)}
    print("\nTask A on TEST:")
    evaluate.compare(res)
    d_m, lv_m = tree_size(mine, True)
    d_s, lv_s = tree_size(sk, False)
    print(f"\n  size: mine depth={d_m} leaves={lv_m} | sklearn depth={d_s} leaves={lv_s}")
    print(f"  fit time:     mine {t_fit_m:.2f}s | sklearn {t_fit_s:.2f}s")
    print(f"  predict time: mine {t_pr_m:.3f}s | sklearn {t_pr_s:.3f}s")
    print(f"  prediction agreement (test): corr={np.corrcoef(pm, ps)[0, 1]:.4f}, "
          f"mean|diff|={np.mean(np.abs(pm - ps)):.4f}, max|diff|={np.max(np.abs(pm - ps)):.4f}")
    print_importances("Task A", mine.feature_importances_, sk.feature_importances_, names)
    for name, m, fit, pr, dep, lv in (("my tree", res["my tree"], t_fit_m, t_pr_m, d_m, lv_m),
                                       ("sklearn", res["sklearn"], t_fit_s, t_pr_s, d_s, lv_s)):
        rows.append(["A", name, f"max_depth={best_d}", *[f"{k}={v:.5f}" for k, v in m.items()],
                     f"depth={dep}", f"leaves={lv}", f"fit_s={fit:.3f}", f"predict_s={pr:.4f}"])

    # ============================ Task B: classification ============================
    for crit in ("gini", "entropy"):
        print(f"\n=== Task B (price tier), criterion={crit}: choose max_depth on VALIDATION ===")
        best_d, best_f = None, -1.0
        for d in DEPTH_GRID:
            m = DecisionTree("classification", crit, max_depth=d, random_state=SEED).fit(X_tr, c_tr)
            f = evaluate.precision_recall_f1(c_va, m.predict(X_va))[2]
            print(f"  max_depth={d:>2}  val F1={f:.4f}")
            if f > best_f:
                best_d, best_f = d, f
        print(f"  -> chosen max_depth={best_d}")

        mine, t_fit_m = timed(lambda: DecisionTree("classification", crit, max_depth=best_d, random_state=SEED).fit(X_tr, c_tr))
        sk, t_fit_s = timed(lambda: DecisionTreeClassifier(criterion=crit, max_depth=best_d, random_state=SEED).fit(X_tr, c_tr))
        (pm, t_pr_m), (ps, t_pr_s) = timed(lambda: mine.predict(X_te)), timed(lambda: sk.predict(X_te))
        prob_m, prob_s = mine.predict_proba(X_te)[:, 1], sk.predict_proba(X_te)[:, 1]

        def cls_metrics(pred, prob):
            p, r, f = evaluate.precision_recall_f1(c_te, pred)
            return {"precision": p, "recall": r, "F1": f, "ROC-AUC": evaluate.roc_auc(c_te, prob),
                    "accuracy": float(np.mean(pred == c_te))}

        res = {"my tree": cls_metrics(pm, prob_m), "sklearn": cls_metrics(ps, prob_s)}
        print(f"\nTask B ({crit}) on TEST:")
        evaluate.compare(res)
        print("  confusion matrix mine   [[TN FP],[FN TP]]:", evaluate.confusion_matrix(c_te, pm).tolist())
        print("  confusion matrix sklearn[[TN FP],[FN TP]]:", evaluate.confusion_matrix(c_te, ps).tolist())
        d_m, lv_m = tree_size(mine, True)
        d_s, lv_s = tree_size(sk, False)
        print(f"  size: mine depth={d_m} leaves={lv_m} | sklearn depth={d_s} leaves={lv_s}")
        print(f"  fit time: mine {t_fit_m:.2f}s | sklearn {t_fit_s:.2f}s ; predict: mine {t_pr_m:.3f}s | sklearn {t_pr_s:.3f}s")
        print(f"  prediction agreement (test): {np.mean(pm == ps):.4f}")
        print_importances(f"Task B ({crit})", mine.feature_importances_, sk.feature_importances_, names)
        for name, m, fit, pr, dep, lv in (("my tree", res["my tree"], t_fit_m, t_pr_m, d_m, lv_m),
                                           ("sklearn", res["sklearn"], t_fit_s, t_pr_s, d_s, lv_s)):
            rows.append([f"B-{crit}", name, f"max_depth={best_d}", *[f"{k}={v:.5f}" for k, v in m.items()],
                         f"depth={dep}", f"leaves={lv}", f"fit_s={fit:.3f}", f"predict_s={pr:.4f}"])

    path = os.path.join(FIG_DIR, "tree_vs_sklearn.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()