import numpy as np
import pytest

from src.unsupervised import (
    fit_segments,
    project_features,
    predict_segments,
)


def sample_data():
    rng = np.random.default_rng(42)
    first = rng.normal(loc=-5.0, scale=0.2, size=(30, 4))
    second = rng.normal(loc=5.0, scale=0.2, size=(30, 4))
    return np.vstack([first, second])


def test_finds_two_separated_groups():
    X = sample_data()
    pipeline = fit_segments(X, n_clusters=2)
    labels = predict_segments(pipeline, X)

    assert np.unique(labels[:30]).size == 1
    assert np.unique(labels[30:]).size == 1
    assert labels[0] != labels[30]


def test_pca_preserves_requested_variance():
    X = sample_data()
    pipeline = fit_segments(X, n_clusters=2, variance=0.90)
    projected = project_features(pipeline, X)
    pca = pipeline.named_steps["pca"]

    assert projected.shape == (len(X), pca.n_components_)
    assert np.isfinite(projected).all()
    assert pca.explained_variance_ratio_.sum() >= 0.90


def test_reproducibility():
    X = sample_data()
    first = fit_segments(X, n_clusters=2, seed=42)
    second = fit_segments(X, n_clusters=2, seed=42)

    np.testing.assert_array_equal(
        predict_segments(first, X),
        predict_segments(second, X),
    )


def test_new_data_does_not_change_fitted_parameters():
    X = sample_data()
    pipeline = fit_segments(X, n_clusters=2)

    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    kmeans = pipeline.named_steps["kmeans"]

    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()
    components_before = pca.components_.copy()
    centers_before = kmeans.cluster_centers_.copy()

    X_new = np.full((5, 4), 1000.0)
    projected = project_features(pipeline, X_new)
    labels = predict_segments(pipeline, X_new)

    assert projected.shape[0] == 5
    assert labels.shape == (5,)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scale_, scale_before)
    np.testing.assert_array_equal(pca.components_, components_before)
    np.testing.assert_array_equal(kmeans.cluster_centers_, centers_before)


@pytest.mark.parametrize(
    "options",
    [
        {"n_clusters": 1},
        {"n_clusters": 100},
        {"variance": 0.0},
        {"variance": 1.0},
    ],
)
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        fit_segments(sample_data(), **options)


def test_invalid_data():
    with pytest.raises(ValueError):
        fit_segments([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        fit_segments([[1.0, np.nan], [2.0, 3.0]], n_clusters=2)