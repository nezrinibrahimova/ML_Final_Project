"""Task B (price tier) with our Pegasos SVM vs scikit-learn baselines.

Run from repo root:  python -m src.svm_experiment

Every hyperparameter is selected on VALIDATION; the test split is scored
once, at the end, for the final configurations only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.data_prep import (NUMERIC_FEATURES, SEED, clean, load_raw, make_features, make_tier_label,
                           standardize, train_val_test_split)
from src.evaluate import compare, confusion_matrix, precision_recall_f1, roc_auc
from src.svm import PegasosSVM, RFFPegasosSVM

FIGURES = Path("report/figures")
TABLES = Path("report/tables")

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
EPOCHS, BATCH = 100, 256          # linear: reaches the LinearSVC objective to <0.5%
RFF_GAMMAS = [0.01, 0.03, 0.1, 0.3]
RFF_LAMBDAS = [1e-6, 1e-5, 1e-4]
RFF_EPOCHS, RFF_BATCH = 30, 64
RFF_SWEEP_D = 1000                # D used while sweeping gamma x lambda
RFF_D_CURVE = [250, 500, 1000, 2000]
RFF_FINAL_D = 2000                # more features = better kernel approximation
SVC_SUBSAMPLE = 10_000     # exact RBF SVC is O(n^2): fit on a stratified subsample
CV_FOLDS = 5


# --- data ---------------------------------------------------------------------
def align_features(df, train_feature_names):
    """One-hot columns defined by the training split; unseen levels -> 0."""
    X, y, names = make_features(df)
    frame = pd.DataFrame(X, columns=names).reindex(columns=train_feature_names, fill_value=0.0)
    return frame.to_numpy(dtype=float), y


def load_tier_splits(df):
    """Same split as the other experiments (price quartiles, seed 42).
    Tier threshold = TRAIN median; all scaling statistics from TRAIN only.

    Two views of the features:
      X_*  : every column standardized (linear models).
      K_*  : numeric columns standardized, 0/1 one-hot columns left as 0/1
             (kernel models). Standardizing a rare district dummy turns its
             1 into ~200, and the RBF distance is then dominated by "which
             rare district" -- val F1 of RFF drops from ~0.92 to ~0.89."""
    prices = df["price"].to_numpy(dtype=float)
    groups = pd.qcut(prices, 4, labels=False)
    i_tr, _, i_va, _, i_te, _ = train_val_test_split(
        np.arange(len(df)), prices, seed=SEED, stratify=groups)

    X_tr, p_tr, names = make_features(df.iloc[i_tr])
    X_va, p_va = align_features(df.iloc[i_va], names)
    X_te, p_te = align_features(df.iloc[i_te], names)

    y_tr, thr = make_tier_label(p_tr)
    y_va, _ = make_tier_label(p_va, thr)
    y_te, _ = make_tier_label(p_te, thr)
    num = len(NUMERIC_FEATURES)
    kernel_view = [np.hstack([z, X[:, num:]])
                   for z, X in zip(standardize(X_tr[:, :num], X_va[:, :num], X_te[:, :num]),
                                   (X_tr, X_va, X_te))]
    X_tr_raw = X_tr
    X_tr, X_va, X_te = standardize(X_tr, X_va, X_te)
    return dict(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te,
                K_tr=kernel_view[0], K_va=kernel_view[1], K_te=kernel_view[2],
                X_tr_raw=X_tr_raw, names=names, threshold=thr)


def cls_metrics(y, pred, scores) -> dict:
    p, r, f1 = precision_recall_f1(y, pred)
    return {"Acc": float((np.asarray(pred) == y).mean()), "Prec": p, "Rec": r,
            "F1": f1, "AUC": roc_auc(y, scores)}


def _timed_fit(model, X, y):
    t = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t


def _timed_scores(model, X):
    t = time.perf_counter()
    s = model.decision_function(X)
    return s, time.perf_counter() - t


def _pegasos(lam, **kw):
    return PegasosSVM(lambda_=lam, n_epochs=EPOCHS, batch_size=BATCH, random_state=SEED, **kw)


def _rff(lam, gamma, D=RFF_SWEEP_D):
    return RFFPegasosSVM(n_components=D, gamma=gamma, lambda_=lam,
                         n_epochs=RFF_EPOCHS, batch_size=RFF_BATCH, random_state=SEED)


# --- sweeps (validation only) -------------------------------------------------
def sweep_linear(d):
    rows = []
    for lam in LAMBDAS:
        m = _pegasos(lam)
        fit_s = _timed_fit(m, d["X_tr"], d["y_tr"])
        s = m.decision_function(d["X_va"])
        met = cls_metrics(d["y_va"], m.predict(d["X_va"]), s)
        rows.append({"lambda": lam, **met, "J_train": m.objective_, "w_norm": m.w_norm_,
                     "margin": m.margin_, "n_support": m.n_support_,
                     "sv_frac": m.n_support_ / len(d["y_tr"]), "fit_s": fit_s})
        print(f"  linear  lambda={lam:<7g} val F1={met['F1']:.4f} AUC={met['AUC']:.4f} "
              f"margin={m.margin_:.3f}  SV={m.n_support_}")
    return pd.DataFrame(rows)


def sweep_rff(d):
    rows = []
    for gamma in RFF_GAMMAS:
        for lam in RFF_LAMBDAS:
            m = _rff(lam, gamma)
            fit_s = _timed_fit(m, d["K_tr"], d["y_tr"])
            s = m.decision_function(d["K_va"])
            met = cls_metrics(d["y_va"], m.predict(d["K_va"]), s)
            rows.append({"gamma": gamma, "lambda": lam, **met, "J_train": m.objective_,
                         "margin": m.margin_, "n_support": m.n_support_, "fit_s": fit_s})
            print(f"  RFF  gamma={gamma:<6g} lambda={lam:<7g} val F1={met['F1']:.4f} "
                  f"AUC={met['AUC']:.4f}  ({fit_s:.1f}s)")
    return pd.DataFrame(rows)


def rff_components_curve(d, lam, gamma):
    """Validation F1/AUC vs number of random features D (kernel approximation quality)."""
    rows = []
    for D in RFF_D_CURVE:
        m = _rff(lam, gamma, D)
        fit_s = _timed_fit(m, d["K_tr"], d["y_tr"])
        met = cls_metrics(d["y_va"], m.predict(d["K_va"]), m.decision_function(d["K_va"]))
        rows.append({"D": D, "F1": met["F1"], "AUC": met["AUC"], "fit_s": fit_s})
        print(f"  RFF  D={D:<5d} val F1={met['F1']:.4f} AUC={met['AUC']:.4f}  ({fit_s:.1f}s)")
    return pd.DataFrame(rows)


def scaling_ablation(d, lam, gamma):
    """RFF at the selected (gamma, lambda) on both feature views (validation)."""
    rows = []
    for view, label in (("X", "all columns standardized"), ("K", "numeric standardized, one-hot 0/1")):
        m = _rff(lam, gamma).fit(d[f"{view}_tr"], d["y_tr"])
        met = cls_metrics(d["y_va"], m.predict(d[f"{view}_va"]), m.decision_function(d[f"{view}_va"]))
        rows.append({"features": label, "max_abs_x": float(np.abs(d[f"{view}_tr"]).max()),
                     "val_F1": met["F1"], "val_AUC": met["AUC"]})
    return pd.DataFrame(rows)


def compare_schedules(d, lam):
    """Objective J(w,b) vs step for the learning-rate schedules, at the chosen lambda."""
    configs = {
        "pegasos, t0=1/lambda (ours)": dict(),
        "pegasos, t0=0 (paper)": dict(t0=0),
        "eta0/sqrt(t), eta0=0.1": dict(lr_schedule="invsqrt", eta0=0.1),
        "constant eta=0.01": dict(lr_schedule="constant", eta0=0.01),
    }
    curves, rows = {}, []
    for name, kw in configs.items():
        m = _pegasos(lam, **kw).fit(d["X_tr"], d["y_tr"])
        curves[name] = (np.array(m.obj_steps_), np.array(m.obj_history_))
        met = cls_metrics(d["y_va"], m.predict(d["X_va"]), m.decision_function(d["X_va"]))
        rows.append({"schedule": name, "J_final": m.objective_, "val_F1": met["F1"],
                     "val_AUC": met["AUC"], "b": m.b})
    return curves, pd.DataFrame(rows)


# --- cross-validation on TRAIN (mean +/- std) ---------------------------------
def cv_scores(make_model, X_raw, y):
    """5-fold stratified CV on the training split. Each fold is standardized
    with its own training-fold stats: sklearn models via Pipeline(StandardScaler),
    our model via data_prep.standardize (it is not an sklearn estimator)."""
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    f1s, aucs, fits = [], [], []
    for tr, va in skf.split(X_raw, y):
        model = make_model()
        X_a, X_b = X_raw[tr], X_raw[va]
        if isinstance(model, PegasosSVM):
            X_a, X_b = standardize(X_a, X_b)
        else:
            model = make_pipeline(StandardScaler(), model)
        fits.append(_timed_fit(model, X_a, y[tr]))
        s = model.decision_function(X_b)
        f1s.append(precision_recall_f1(y[va], model.predict(X_b))[2])
        aucs.append(roc_auc(y[va], s))
    return {"CV F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "CV AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            "CV fit s": float(np.mean(fits))}


def stratified_subsample(y, n, seed=SEED):
    rng = np.random.default_rng(seed)
    if n >= len(y):
        return np.arange(len(y))
    idx = [rng.choice(np.flatnonzero(y == c), int(round(n * (y == c).mean())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


# --- figures -------------------------------------------------------------------
def plot_objective(curves):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, (steps, J) in curves.items():
        ax.plot(steps, J, label=name)
    ax.set_xlabel("sub-gradient step t")
    ax.set_ylabel("J(w, b) on train")
    ax.set_yscale("log")
    ax.set_title("Pegasos objective during training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_objective.png", dpi=200)
    plt.close(fig)


def plot_lambda_sweep(lin):
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.semilogx(lin["lambda"], lin["F1"], "o-", label="val F1")
    ax1.semilogx(lin["lambda"], lin["AUC"], "s-", label="val AUC")
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("validation score")
    ax2 = ax1.twinx()
    ax2.semilogx(lin["lambda"], lin["margin"], "^--", color="tab:gray", label="margin 2/||w||")
    ax2.set_yscale("log")
    ax2.set_ylabel("margin 2/||w||")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("Linear Pegasos: lambda vs validation score and margin")
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_lambda_sweep.png", dpi=200)
    plt.close(fig)


def plot_margin_tradeoff(lin):
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(lin["margin"], lin["F1"], c=np.log10(lin["lambda"]), cmap="viridis", s=60)
    for _, r in lin.iterrows():
        ax.annotate(f"{r['sv_frac']:.0%} SV", (r["margin"], r["F1"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("margin 2/||w||")
    ax.set_ylabel("validation F1")
    fig.colorbar(sc, label="log10 lambda")
    ax.set_title("Wider margin vs fit (labels: support-vector share)")
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_margin_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_rff_heatmap(rff):
    piv = rff.pivot(index="gamma", columns="lambda", values="F1")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="w")
    ax.set_xlabel("lambda")
    ax.set_ylabel("gamma (RBF bandwidth)")
    ax.set_title(f"RFF Pegasos (D={RFF_SWEEP_D}): validation F1")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_rff_heatmap.png", dpi=200)
    plt.close(fig)


def plot_rff_components(curve):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogx(curve["D"], curve["F1"], "o-", label="val F1")
    ax.semilogx(curve["D"], curve["AUC"], "s-", label="val AUC")
    ax.set_xticks(curve["D"], [str(v) for v in curve["D"]])
    ax.set_xlabel("number of random Fourier features D")
    ax.set_ylabel("validation score")
    ax.set_title("RFF Pegasos: kernel approximation vs D")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_rff_components.png", dpi=200)
    plt.close(fig)


def plot_top_weights(model, names, k=15):
    w = model.w
    top = np.argsort(np.abs(w))[::-1][:k][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * k + 1))
    ax.barh([names[i] for i in top], w[top], color=np.where(w[top] > 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("weight on standardized feature (+ pushes towards premium)")
    ax.set_title(f"Linear Pegasos: top {k} |w|")
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_top_weights.png", dpi=200)
    plt.close(fig)


# --- main ------------------------------------------------------------------------
def run(df) -> dict:
    """Run the full Task-B SVM study on a cleaned frame; return test metrics per model."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    d = load_tier_splits(df)
    n_tr = len(d["y_tr"])
    print(f"Task B rows: train={n_tr}, val={len(d['y_va'])}, test={len(d['y_te'])}; "
          f"features={len(d['names'])}; tier threshold (train median)={d['threshold']:,.0f} AZN; "
          f"premium share train={d['y_tr'].mean():.3f}")

    print("\nLinear Pegasos lambda sweep (validation):")
    lin = sweep_linear(d)
    lam = float(lin.loc[lin["F1"].idxmax(), "lambda"])
    print(f"-> selected lambda={lam:g}")

    print("\nRFF Pegasos gamma x lambda sweep (validation):")
    rff = sweep_rff(d)
    best = rff.loc[rff["F1"].idxmax()]
    r_lam, r_gamma = float(best["lambda"]), float(best["gamma"])
    print(f"-> selected gamma={r_gamma:g}, lambda={r_lam:g}")
    print("\nRFF Pegasos vs D at the selected gamma, lambda (validation):")
    d_curve = rff_components_curve(d, r_lam, r_gamma)
    scaling = scaling_ablation(d, r_lam, r_gamma)
    print("\nRFF feature-scaling ablation (validation):")
    print(scaling.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    curves, sched = compare_schedules(d, lam)
    print("\nLearning-rate schedules at the selected lambda:")
    print(sched.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- final models: fit on train, score test once ----------------------------
    sub = stratified_subsample(d["y_tr"], SVC_SUBSAMPLE)
    models = {   # name: (model, feature view, training rows)
        "Pegasos linear (ours)": (_pegasos(lam), "X", slice(None)),
        "Pegasos + RFF (ours)": (_rff(r_lam, r_gamma, RFF_FINAL_D), "K", slice(None)),
        "sklearn LinearSVC": (LinearSVC(C=1 / (lam * n_tr), loss="hinge", max_iter=50_000,
                                        random_state=SEED), "X", slice(None)),
        "sklearn SGDClassifier": (SGDClassifier(loss="hinge", alpha=lam, max_iter=EPOCHS,
                                                tol=None, random_state=SEED), "X", slice(None)),
        f"sklearn SVC rbf (n={len(sub)})": (SVC(kernel="rbf", gamma=r_gamma,
                                               C=1 / (r_lam * len(sub))), "K", sub),
    }
    results, timing = {}, []
    for name, (m, view, rows) in models.items():
        fit_s = _timed_fit(m, d[f"{view}_tr"][rows], d["y_tr"][rows])
        s, pred_s = _timed_scores(m, d[f"{view}_te"])
        pred = m.predict(d[f"{view}_te"])
        results[name] = {**cls_metrics(d["y_te"], pred, s), "fit s": fit_s, "predict s": pred_s}
        cm = confusion_matrix(d["y_te"], pred)
        timing.append({"model": name, "TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1]})
    final_linear = models["Pegasos linear (ours)"][0]
    final_rff = models["Pegasos + RFF (ours)"][0]

    # --- CV mean +/- std on train (same hyperparameters) ------------------------
    print(f"\n{CV_FOLDS}-fold CV on train (scaler inside the pipeline):")
    Xr, yr = d["X_tr_raw"], d["y_tr"]
    cv_models = {
        "Pegasos linear (ours)": lambda: _pegasos(lam),
        "sklearn LinearSVC": lambda: LinearSVC(C=1 / (lam * n_tr * (CV_FOLDS - 1) / CV_FOLDS),
                                               loss="hinge", max_iter=50_000, random_state=SEED),
        "sklearn SGDClassifier": lambda: SGDClassifier(loss="hinge", alpha=lam, max_iter=EPOCHS,
                                                       tol=None, random_state=SEED),
    }
    cv = {}
    for name, make in cv_models.items():
        cv[name] = cv_scores(make, Xr, yr)
        print(f"  {name:<24} F1 {cv[name]['CV F1']}   AUC {cv[name]['CV AUC']}")

    # --- tables + figures --------------------------------------------------------
    lin.to_csv(TABLES / "svm_lambda_sweep.csv", index=False)
    rff.to_csv(TABLES / "svm_rff_sweep.csv", index=False)
    d_curve.to_csv(TABLES / "svm_rff_components.csv", index=False)
    scaling.to_csv(TABLES / "svm_rff_scaling.csv", index=False)
    sched.to_csv(TABLES / "svm_schedules.csv", index=False)
    pd.DataFrame(results).T.to_csv(TABLES / "svm_task_b_results.csv")
    pd.DataFrame(timing).to_csv(TABLES / "svm_confusion.csv", index=False)
    pd.DataFrame(cv).T.to_csv(TABLES / "svm_cv.csv")
    w = pd.Series(final_linear.w, index=d["names"])
    w.reindex(w.abs().sort_values(ascending=False).index).to_csv(TABLES / "svm_weights.csv", header=["w"])

    plot_objective(curves)
    plot_lambda_sweep(lin)
    plot_margin_tradeoff(lin)
    plot_rff_heatmap(rff)
    plot_rff_components(d_curve)
    plot_top_weights(final_linear, d["names"])

    print(f"\nFinal linear Pegasos: ||w||={final_linear.w_norm_:.4f}, margin={final_linear.margin_:.4f}, "
          f"support vectors={final_linear.n_support_}/{n_tr}, J={final_linear.objective_:.4f}")
    print(f"Final RFF Pegasos:    ||w||={final_rff.w_norm_:.4f}, margin={final_rff.margin_:.4f}, "
          f"support vectors={final_rff.n_support_}/{n_tr}")
    print("\nTask B test results (SVM family):")
    compare(results)
    print("\nSaved SVM figures to report/figures and tables to report/tables.")
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles: "±", Azerbaijani names
    run(clean(load_raw()))


if __name__ == "__main__":
    main()
