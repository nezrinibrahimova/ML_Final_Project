"""Housing segments and validation error analysis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_prep import (
    load_raw, clean, make_features, train_val_test_split,
)
from src.unsupervised import (
    fit_segments, project_features, predict_segments,
)


def main():
    df = clean(load_raw())
    prices = df["price"].to_numpy(dtype=float)
    indices = np.arange(len(df))
    groups = pd.qcut(prices, 4, labels=False)

    i_train, _, i_val, _, i_test, _ = train_val_test_split(
        indices, prices, seed=42, stratify=groups
    )

    train = df.iloc[i_train].copy()
    val = df.iloc[i_val].copy()

    X_train, y_train, names = make_features(train)
    X_val, y_val, val_names = make_features(val)

    # Keep the training feature vocabulary and column order.
    X_val = (
        pd.DataFrame(X_val, columns=val_names)
        .reindex(columns=names, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    # Five clusters are an exploratory choice, not a tuned optimum.
    segments = fit_segments(
        X_train, n_clusters=5, variance=0.90, seed=42
    )
    train_labels = predict_segments(segments, X_train)
    val_labels = predict_segments(segments, X_val)
    projected = project_features(segments, X_train)
    pca = segments.named_steps["pca"]

    print(
        f"Rows: train={len(train)}, validation={len(val)}, "
        f"test={len(i_test)} (held out)"
    )
    print(f"PCA components: {pca.n_components_}")
    print(
        "Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    # Prices are used only to describe clusters after fitting.
    train["cluster"] = train_labels
    summary = train.groupby("cluster").agg(
        count=("price", "size"),
        median_price=("price", "median"),
        median_area=("area_m2", "median"),
        median_rooms=("rooms", "median"),
        median_lat=("lat", "median"),
        median_lng=("lng", "median"),
    )

    print("\nTraining cluster summary:")
    print(summary.round(2).to_string())

    # A fixed sklearn baseline for exploratory error analysis.
    # This is separate from our from-scratch Ridge experiment.
    baseline = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, solver="cholesky"),
    )
    baseline.fit(X_train, y_train)
    predictions = baseline.predict(X_val)
    residuals = predictions - y_val

    errors = pd.DataFrame({
        "cluster": val_labels,
        "absolute_error": np.abs(residuals),
        "squared_error": residuals ** 2,
        "signed_error": residuals,
    })
    error_summary = errors.groupby("cluster").agg(
        count=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        MSE=("squared_error", "mean"),
        bias=("signed_error", "mean"),
    )
    error_summary["RMSE"] = np.sqrt(error_summary.pop("MSE"))
    error_summary = error_summary[["count", "MAE", "RMSE", "bias"]]

    print("\nValidation errors by cluster — sklearn Ridge, alpha=1:")
    print(error_summary.round(2).to_string())

    figures = Path("report/figures")
    tables = Path("report/tables")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summary.to_csv(tables / "cluster_train_summary.csv")
    # Geographic view of clusters learned from training features.
    sample = train.sample(n=min(10000, len(train)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in sorted(train["cluster"].unique()):
        points = sample[sample["cluster"] == cluster]
        ax.scatter(
            points["lng"],
            points["lat"],
            s=10,
            alpha=0.45,
            label=f"Cluster {cluster}",
        )

    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Training Clusters — Geographic Distribution",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "cluster_geography.png", dpi=200)
    plt.close(fig)

    # Describe prices after clustering; price was not a clustering feature.
    mean_prices = (
        train.groupby("cluster")["price"]
        .mean()
        .sort_index()
        .rename("mean_price_AZN")
    )
    mean_prices.to_csv(tables / "cluster_mean_prices.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        mean_prices.index.astype(str),
        mean_prices.to_numpy(),
    )
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(
        xlabel="Cluster",
        ylabel="Mean Price (AZN)",
        title="Mean Housing Price by Training Cluster",
    )
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(figures / "cluster_mean_prices.png", dpi=200)
    plt.close(fig)
    error_summary.to_csv(tables / "cluster_validation_errors.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".")
    ax.axhline(0.90, color="red", linestyle="--", label="90%")
    ax.set(
        xlabel="Number of retained components",
        ylabel="Cumulative explained variance",
        title="PCA explained variance — training data",
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "pca_variance.png", dpi=200)
    plt.close(fig)

    rng = np.random.default_rng(42)
    sample = rng.choice(
        len(X_train), size=min(5000, len(X_train)), replace=False
    )
    second_axis = (
        projected[:, 1]
        if projected.shape[1] >= 2
        else np.zeros(len(projected))
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in range(5):
        selected = sample[train_labels[sample] == cluster]
        ax.scatter(
            projected[selected, 0],
            second_axis[selected],
            s=10, alpha=0.4, label=f"Cluster {cluster}",
        )
    ax.set(
        xlabel="Principal component 1",
        ylabel=(
            "Principal component 2"
            if projected.shape[1] >= 2
            else "Zero placeholder (only one component retained)"
        ),
        title="Training segments — PCA projection",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "pca_clusters.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        error_summary.index.astype(str),
        error_summary["MAE"],
    )
    ax.set(
        xlabel="Cluster",
        ylabel="Validation MAE (AZN)",
        title="Sklearn Ridge baseline error by segment",
    )
    fig.tight_layout()
    fig.savefig(figures / "cluster_validation_mae.png", dpi=200)
    plt.close(fig)

    print("\nSaved figures to report/figures and tables to report/tables.")


if __name__ == "__main__":
    main()