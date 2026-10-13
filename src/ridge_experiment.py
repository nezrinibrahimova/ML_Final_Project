"""Run the Ridge experiment with a fixed train/validation/test split."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data_prep import (
    load_raw,
    clean,
    make_features,
    train_val_test_split,
    standardize,
)
from src.evaluate import rmse, mae, r2, compare
from src.ridge import RidgeRegression


def align_features(df, train_feature_names):
    """Use only the feature columns defined by the training set."""
    X, y, names = make_features(df)
    frame = pd.DataFrame(X, columns=names)
    frame = frame.reindex(columns=train_feature_names, fill_value=0.0)
    return frame.to_numpy(dtype=float), y


def metrics(y, predictions):
    return {
        "RMSE": rmse(y, predictions),
        "MAE": mae(y, predictions),
        "R2": r2(y, predictions),
    }


def main():
    df = clean(load_raw())
    prices = df["price"].to_numpy(dtype=float)
    indices = np.arange(len(df))
    groups = pd.qcut(prices, 4, labels=False)

    i_train, _, i_val, _, i_test, _ = train_val_test_split(
        indices, prices, seed=42, stratify=groups
    )

    train = df.iloc[i_train]
    val = df.iloc[i_val]
    test = df.iloc[i_test]

    X_train, y_train, names = make_features(train)
    X_val, y_val = align_features(val, names)
    X_test, y_test = align_features(test, names)

    X_train, X_val, X_test = standardize(
        X_train, X_val, X_test
    )

    print(
        f"Rows: train={len(y_train)}, "
        f"validation={len(y_val)}, test={len(y_test)}"
    )

    best_model = None
    best_alpha = None
    best_val_rmse = float("inf")

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = RidgeRegression(alpha=alpha).fit(X_train, y_train)
        score = rmse(y_val, model.predict(X_val))
        print(f"alpha={alpha:g}  validation RMSE={score:,.2f}")

        if score < best_val_rmse:
            best_val_rmse = score
            best_alpha = alpha
            best_model = model

    print(f"\nSelected alpha: {best_alpha:g}")

    baseline = Ridge(
        alpha=best_alpha, solver="cholesky"
    ).fit(X_train, y_train)

    # Evaluate the selected model on the test set only after selection.
    our_predictions = best_model.predict(X_test)
    baseline_predictions = baseline.predict(X_test)
    mean_predictions = np.full(len(y_test), y_train.mean())

    print("\nTest results:")
    compare({
        "Train-mean baseline": metrics(y_test, mean_predictions),
        "NumPy Ridge": metrics(y_test, our_predictions),
        "sklearn Ridge": metrics(y_test, baseline_predictions),
    })

    difference = np.max(
        np.abs(our_predictions - baseline_predictions)
    )
    print(f"\nMaximum prediction difference: {difference:.8f}")


if __name__ == "__main__":
    main()