"""A-part: matched Ridge/Tree comparison and cluster error analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_prep import (
    clean, load_raw, make_features,
    train_val_test_split, standardize,
)
from src.decision_tree import DecisionTree
from src.evaluate import rmse, mae, r2, precision_recall_f1, roc_auc
from src.ridge import RidgeRegression
from src.svm import PegasosSVM
from src.unsupervised import fit_segments, predict_segments


SEED = 42
TABLES = Path("report/tables")


def aligned_features(df, names):
    X, y, current_names = make_features(df)
    X = (
        pd.DataFrame(X, columns=current_names)
        .reindex(columns=names, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    return X, y


def regression_metrics(y, pred):
    return {
        "RMSE": rmse(y, pred),
        "MAE": mae(y, pred),
        "R2": r2(y, pred),
    }


def classification_metrics(y, pred, scores):
    precision, recall, f1 = precision_recall_f1(y, pred)
    return {
        "accuracy": float(np.mean(y == pred)),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "AUC": roc_auc(y, scores) if np.unique(y).size == 2 else np.nan,
    }


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    prices = df["price"].to_numpy(dtype=float)

    # Same split convention as Ridge, SVM and PCA experiments.
    groups = pd.qcut(prices, 4, labels=False)
    train_idx, _, val_idx, _, test_idx, _ = train_val_test_split(
        np.arange(len(df)), prices, seed=SEED, stratify=groups
    )

    X_train, y_train, names = make_features(df.iloc[train_idx])
    X_val, y_val = aligned_features(df.iloc[val_idx], names)
    X_test, y_test = aligned_features(df.iloc[test_idx], names)

    Z_train, Z_val, Z_test = standardize(X_train, X_val, X_test)
    threshold = float(np.median(y_train))
    c_train = (y_train > threshold).astype(int)
    c_val = (y_val > threshold).astype(int)

    print(
        f"Rows: {len(y_train)} / {len(y_val)} / {len(y_test)}",
        flush=True,
    )

    # 1. Ridge: select alpha using validation RMSE in AZN.
    best_ridge, best_alpha = None, None
    best_score = float("inf")
    selection = []

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = RidgeRegression(alpha=alpha).fit(Z_train, y_train)
        score = rmse(y_val, model.predict(Z_val))
        selection.append({
            "model": "Ridge", "parameter": f"alpha={alpha}",
            "metric": "RMSE_AZN", "validation_score": score,
        })
        if score < best_score:
            best_ridge, best_alpha, best_score = model, alpha, score

    print(f"Ridge selected alpha={best_alpha}", flush=True)

    # 2. Tree regression: match B's log-price training objective.
    # Select depth by validation log-RMSE; report final errors in AZN.
    best_tree, best_depth = None, None
    best_score = float("inf")

    for depth in [6, 8, 10, 12, 15]:
        model = DecisionTree(
            task="regression", criterion="mse",
            max_depth=depth, random_state=SEED,
        ).fit(X_train, np.log(y_train))
        score = rmse(np.log(y_val), model.predict(X_val))
        print(f"Tree regression depth={depth}: {score:.5f}", flush=True)
        selection.append({
            "model": "Tree regression", "parameter": f"depth={depth}",
            "metric": "RMSE_log", "validation_score": score,
        })
        if score < best_score:
            best_tree, best_depth, best_score = model, depth, score

    # Test set is used after both regression configurations are selected.
    comparison = pd.DataFrame.from_dict({
        "Train-mean baseline": regression_metrics(
            y_test, np.full(len(y_test), y_train.mean())
        ),
        "NumPy Ridge": regression_metrics(
            y_test, best_ridge.predict(Z_test)
        ),
        "NumPy Tree (trained on log-price)": regression_metrics(
            y_test, np.exp(best_tree.predict(X_test))
        ),
    }, orient="index")
    comparison.index.name = "model"
    comparison.to_csv(TABLES / "a_matched_task_a_results.csv")
    print("\nMatched TEST comparison (AZN):")
    print(comparison.round(4).to_string(), flush=True)

    # 3. Tree classification: select criterion/depth by validation F1.
    best_classifier, best_cls_config = None, None
    best_f1 = -1.0

    for criterion in ["gini", "entropy"]:
        for depth in [6, 8, 10, 12, 15]:
            model = DecisionTree(
                task="classification", criterion=criterion,
                max_depth=depth, random_state=SEED,
            ).fit(X_train, c_train)
            score = precision_recall_f1(
                c_val, model.predict(X_val)
            )[2]
            print(
                f"Tree classification {criterion}, depth={depth}: "
                f"F1={score:.5f}", flush=True,
            )
            selection.append({
                "model": "Tree classification",
                "parameter": f"{criterion}, depth={depth}",
                "metric": "F1", "validation_score": score,
            })
            if score > best_f1:
                best_classifier = model
                best_cls_config = {"criterion": criterion, "depth": depth}
                best_f1 = score

    # 4. Linear Pegasos: same grid/training settings as C's experiment.
    best_svm, best_lambda = None, None
    best_f1 = -1.0

    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        model = PegasosSVM(
            lambda_=lam, n_epochs=100, batch_size=256,
            random_state=SEED,
        ).fit(Z_train, c_train)
        score = precision_recall_f1(c_val, model.predict(Z_val))[2]
        print(f"Linear SVM lambda={lam}: F1={score:.5f}", flush=True)
        selection.append({
            "model": "Linear Pegasos", "parameter": f"lambda={lam}",
            "metric": "F1", "validation_score": score,
        })
        if score > best_f1:
            best_svm, best_lambda, best_f1 = model, lam, score

    # 5. Fit clusters on TRAIN features only, without price.
    print("\nFitting PCA/K-means...", flush=True)
    segmenter = fit_segments(
        X_train, n_clusters=5, variance=0.90, seed=SEED
    )
    val_clusters = predict_segments(segmenter, X_val)

    regression_predictions = {
        "NumPy Ridge": best_ridge.predict(Z_val),
        "NumPy Tree": np.exp(best_tree.predict(X_val)),
    }
    classification_predictions = {
        "NumPy Tree": (
            best_classifier.predict(X_val),
            best_classifier.predict_proba(X_val)[:, 1],
        ),
        "Linear Pegasos": (
            best_svm.predict(Z_val),
            best_svm.decision_function(Z_val),
        ),
    }

    reg_rows, cls_rows = [], []
    for cluster in np.unique(val_clusters):
        mask = val_clusters == cluster
        for name, pred in regression_predictions.items():
            reg_rows.append({
                "cluster": int(cluster), "model": name,
                "count": int(mask.sum()),
                **regression_metrics(y_val[mask], pred[mask]),
                "bias": float(np.mean(pred[mask] - y_val[mask])),
            })

        for name, (pred, scores) in classification_predictions.items():
            cls_rows.append({
                "cluster": int(cluster), "model": name,
                "count": int(mask.sum()),
                "premium_count": int(c_val[mask].sum()),
                **classification_metrics(
                    c_val[mask], pred[mask], scores[mask]
                ),
            })

    regression_clusters = pd.DataFrame(reg_rows)
    classification_clusters = pd.DataFrame(cls_rows)

    regression_clusters.to_csv(
        TABLES / "a_cluster_validation_regression.csv", index=False
    )
    classification_clusters.to_csv(
        TABLES / "a_cluster_validation_classification.csv", index=False
    )
    pd.DataFrame(selection).to_csv(
        TABLES / "a_validation_selection.csv", index=False
    )

    protocol = {
        "seed": SEED,
        "split": "pd.qcut(price, 4), 70/15/15",
        "rows": [len(y_train), len(y_val), len(y_test)],
        "features": len(names),
        "tier_threshold_train": threshold,
        "ridge_alpha": best_alpha,
        "tree_regression_depth": best_depth,
        "tree_regression_target": "log(price)",
        "tree_regression_selection": "validation log-RMSE",
        "tree_classification": best_cls_config,
        "linear_svm_lambda": best_lambda,
        "clusters": 5,
        "pca_components": int(segmenter.named_steps["pca"].n_components_),
        "cluster_analysis_split": "validation",
        "note": (
            "Cluster metrics are exploratory validation diagnostics "
            "on the same set used for hyperparameter selection; "
            "they are not independent test estimates. "
            "SVM analysis covers linear Pegasos, not RFF."
        ),
    }
    (TABLES / "a_analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\nCluster regression errors — VALIDATION:")
    print(regression_clusters.round(4).to_string(index=False))
    print("\nCluster classification errors — VALIDATION:")
    print(classification_clusters.round(4).to_string(index=False))
    print("\nDone. Results saved under report/tables/.")


if __name__ == "__main__":
    # Avoid excessive CPU threading overhead on small matrix operations.
    with threadpool_limits(limits=1):
        main()