"""PCA and K-means for housing market segmentation."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_segments(X_train, n_clusters=5, variance=0.90, seed=42):
    """Fit scaling, PCA and clustering using training features only."""
    X_train = np.asarray(X_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if X_train.shape[0] < 2 or X_train.shape[1] == 0:
        raise ValueError("Need at least two rows and one feature")
    if not np.isfinite(X_train).all():
        raise ValueError("X_train must contain only finite values")
    if not 0 < variance < 1:
        raise ValueError("variance must be between 0 and 1")
    if not isinstance(n_clusters, int) or not 2 <= n_clusters <= len(X_train):
        raise ValueError("n_clusters must be an integer from 2 to n_samples")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance, svd_solver="full")),
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=seed,
        )),
    ])

    pipeline.fit(X_train)
    return pipeline


def project_features(pipeline, X):
    """Apply the fitted scaler and PCA without fitting again."""
    X = np.asarray(X, dtype=float)
    return pipeline[:-1].transform(X)


def predict_segments(pipeline, X):
    """Assign new rows to the clusters learned from training data."""
    X = np.asarray(X, dtype=float)
    return pipeline.predict(X)