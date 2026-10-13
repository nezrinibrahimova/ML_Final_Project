import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.ridge import RidgeRegression


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_matches_sklearn(alpha):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4)) + 3.0
    y = X @ np.array([2.0, -1.0, 0.5, 4.0]) + 7.0
    y += rng.normal(scale=0.2, size=80)
    X_test = rng.normal(size=(20, 4)) + 3.0

    ours = RidgeRegression(alpha=alpha).fit(X, y)
    baseline = Ridge(alpha=alpha, solver="cholesky").fit(X, y)

    np.testing.assert_allclose(
        ours.coef_, baseline.coef_, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.intercept_, baseline.intercept_, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.predict(X_test),
        baseline.predict(X_test),
        rtol=1e-8,
        atol=1e-8,
    )


def test_constant_target_with_duplicate_features():
    x = np.arange(10, dtype=float)
    X = np.column_stack([x, x, np.ones(10)])
    y = np.full(10, 25.0)

    model = RidgeRegression(alpha=2.0).fit(X, y)

    np.testing.assert_allclose(model.coef_, 0.0, atol=1e-10)
    assert np.isclose(model.intercept_, 25.0)
    np.testing.assert_allclose(model.predict(X), y)


def test_predict_before_fit():
    with pytest.raises(ValueError):
        RidgeRegression().predict([[1.0, 2.0]])


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        RidgeRegression(alpha=alpha)


def test_invalid_inputs():
    model = RidgeRegression()

    with pytest.raises(ValueError):
        model.fit([[1.0], [2.0]], [1.0])

    with pytest.raises(ValueError):
        model.fit([[1.0], [np.nan]], [1.0, 2.0])

    model.fit([[1.0], [2.0]], [2.0, 4.0])

    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])