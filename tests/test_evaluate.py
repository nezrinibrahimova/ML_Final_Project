import numpy as np
import pytest

from src.evaluate import (
    rmse, mae, r2,
    confusion_matrix, precision_recall_f1,
    roc_auc, compare
)


def test_regression_metrics():
    y = [100, 200, 300]
    p = [110, 190, 330]

    assert np.isclose(rmse(y, p), 19.1485421551)
    assert np.isclose(mae(y, p), 16.6666666667)
    assert np.isclose(r2(y, p), 0.945)


def test_classification_metrics():
    y = [1, 1, 1, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0]

    np.testing.assert_array_equal(
        confusion_matrix(y, p),
        [[2, 1], [1, 2]]
    )

    precision, recall, f1 = precision_recall_f1(y, p)
    assert np.isclose(precision, 2 / 3)
    assert np.isclose(recall, 2 / 3)
    assert np.isclose(f1, 2 / 3)


def test_roc_auc():
    y = [1, 0, 1, 0]
    scores = [0.9, 0.2, 0.7, 0.4]

    assert np.isclose(roc_auc(y, scores), 1.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        rmse([], [])

    with pytest.raises(ValueError):
        rmse([1, np.nan], [1, 2])

    with pytest.raises(ValueError):
        confusion_matrix([0, 0.7], [0, 1])

    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.2, np.inf])


def test_compare(capsys):
    compare({
        "Our Tree": {"RMSE": 45000, "R2": 0.82},
        "Baseline": {"RMSE": 42000, "R2": 0.85}
    })

    output = capsys.readouterr().out
    assert "Our Tree" in output
    assert "Baseline" in output
    assert "RMSE" in output

def test_r2_constant_target():
    # Eyni qiymətləri tam düzgün proqnozlaşdırır.
    assert r2([5, 5, 5], [5, 5, 5]) == 1.0

    # Həqiqi qiymətlər eynidir, amma proqnozda səhv var.
    assert r2([5, 5, 5], [5, 6, 5]) == 0.0