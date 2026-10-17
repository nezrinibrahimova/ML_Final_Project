from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate
from src.data_prep import SEED, clean, load_raw, make_features, make_tier_label, train_val_test_split
from src.decision_tree import DecisionTree

FIGURES = Path("report/figures")
TABLES = Path("report/tables")
DEPTH_GRID = [10, 12, 15, 20]
LEAF_GRID = [10, 5, 3, 1]          # simplest first: on ties the simpler tree wins
_CONFIG_CACHE: dict = {}


def utf8_console():
    """Windows consoles (cp1254/cp1252) cannot print Azerbaijani letters or '±'."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def align_features(df, train_feature_names):
    """One-hot columns defined by the training split; unseen levels -> 0."""
    X, y, names = make_features(df)
    frame = pd.DataFrame(X, columns=names).reindex(columns=train_feature_names, fill_value=0.0)
    return frame.to_numpy(dtype=float), y


def load_splits(synthetic: bool = False) -> dict:
    """dict with X_tr/X_va/X_te, p_* (price), c_* (tier 0/1), names, threshold."""
    if synthetic:
        rng = np.random.default_rng(SEED)
        n = 6000
        X = rng.normal(size=(n, 8))
        price = np.exp(11 + 0.8 * X[:, 0] + 0.4 * np.sin(2 * X[:, 1]) + 0.3 * X[:, 2] * X[:, 3]
                       + rng.normal(0, .3, n))
        idx = rng.permutation(n)
        a, b = int(.7 * n), int(.85 * n)
        tr, va, te = idx[:a], idx[a:b], idx[b:]
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        p_tr, p_va, p_te = price[tr], price[va], price[te]
        names = [f"f{i}" for i in range(8)]
    else:
        df = clean(load_raw())
        prices = df["price"].to_numpy(dtype=float)
        groups = pd.qcut(prices, 4, labels=False)
        i_tr, _, i_va, _, i_te, _ = train_val_test_split(
            np.arange(len(df)), prices, seed=SEED, stratify=groups)
        X_tr, p_tr, names = make_features(df.iloc[i_tr])
        X_va, p_va = align_features(df.iloc[i_va], names)
        X_te, p_te = align_features(df.iloc[i_te], names)
    c_tr, thr = make_tier_label(p_tr)
    c_va, _ = make_tier_label(p_va, thr)
    c_te, _ = make_tier_label(p_te, thr)
    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te, p_tr=p_tr, p_va=p_va, p_te=p_te,
                c_tr=c_tr, c_va=c_va, c_te=c_te, names=names, threshold=thr)


def select_config(d: dict, verbose: bool = True) -> dict:
    """Final hyperparameters per model, chosen on VALIDATION ONLY with my tree over the
    joint grid max_depth x min_samples_leaf. Keys: 'A', 'B-gini', 'B-entropy'; each value is
    {"max_depth": .., "min_samples_leaf": ..}. The choice is cached in memory and in
    report/tables/tree_selected_config.json (re-used only if data and grids are unchanged),
    so every tree script in the project uses exactly the same configuration."""
    key = json.dumps([list(d["X_tr"].shape), round(float(d["p_tr"].sum()), 3), DEPTH_GRID, LEAF_GRID])
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    path = TABLES / "tree_selected_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            if verbose:
                print(f"Using saved validation choice from {path}: {saved['config']}")
            _CONFIG_CACHE[key] = saved["config"]
            return saved["config"]

    l_tr, l_va = np.log(d["p_tr"]), np.log(d["p_va"])
    out, val_scores = {}, {}
    if verbose:
        print("\nChoosing max_depth x min_samples_leaf on VALIDATION (my tree)")
    for name, task, crit in (("A", "regression", "mse"), ("B-gini", "classification", "gini"),
                             ("B-entropy", "classification", "entropy")):
        best, best_score = None, None
        for leaf in LEAF_GRID:
            for dep in DEPTH_GRID:
                if task == "regression":
                    m = DecisionTree(task, crit, max_depth=dep, min_samples_leaf=leaf, random_state=SEED).fit(d["X_tr"], l_tr)
                    score = -evaluate.rmse(l_va, m.predict(d["X_va"]))          # higher is better
                else:
                    m = DecisionTree(task, crit, max_depth=dep, min_samples_leaf=leaf, random_state=SEED).fit(d["X_tr"], d["c_tr"])
                    score = evaluate.precision_recall_f1(d["c_va"], m.predict(d["X_va"]))[2]
                if verbose:
                    shown = f"RMSE(log)={-score:.4f}" if task == "regression" else f"F1={score:.4f}"
                    print(f"  {name:<9} depth={dep:>2} min_leaf={leaf:>2}  val {shown}")
                if best_score is None or score > best_score + 1e-12:
                    best, best_score = {"max_depth": dep, "min_samples_leaf": leaf}, score
        out[name] = best
        val_scores[name] = -best_score if task == "regression" else best_score
    if verbose:
        print(f"  -> chosen: {out}")
    TABLES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "config": out, "validation_score": val_scores}, indent=2),
                    encoding="utf-8")
    _CONFIG_CACHE[key] = out
    return out
