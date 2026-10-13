import numpy as np
import pandas as pd

from src.data_prep import (
    clean,
    load_raw,
    make_features,
    make_tier_label,
    train_val_test_split,
    standardize,
)


def test_load_and_clean():
    df = load_raw()
    cleaned = clean(df)

    assert len(df) == 100775
    assert len(cleaned) in (58008, 58007)
    assert cleaned.isna().sum().sum() == 0
    assert "total_price" not in cleaned.columns
    assert "unit_price" not in cleaned.columns


def test_make_features():
    df = clean(load_raw())
    X, y, feature_names = make_features(df)

    assert X.shape[0] == len(y)
    assert X.shape[1] == len(feature_names)
    assert X.shape[1] == 151
    assert "price" not in feature_names
    assert np.isfinite(X).all()


def test_make_tier_label():
    train_prices = np.array([100, 200, 300, 400])
    labels, threshold = make_tier_label(train_prices)

    assert threshold == 250
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])

    test_labels, _ = make_tier_label([150, 350], threshold)
    np.testing.assert_array_equal(test_labels, [0, 1])


def test_split_reproducibility():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)

    first = train_val_test_split(X, y, seed=42)
    second = train_val_test_split(X, y, seed=42)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)

    assert len(first[1]) + len(first[3]) + len(first[5]) == 100


def test_standardize():
    X_tr = np.array([[1., 10.], [3., 20.], [5., 30.]])
    X_te = np.array([[7., 40.]])

    tr_scaled, te_scaled = standardize(X_tr, X_te)

    np.testing.assert_allclose(tr_scaled.mean(axis=0), [0, 0], atol=1e-10)
    np.testing.assert_allclose(tr_scaled.std(axis=0), [1, 1])
    np.testing.assert_allclose(
        te_scaled,
        (X_te - X_tr.mean(axis=0)) / X_tr.std(axis=0)
    )