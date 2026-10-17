import numpy as np
import pandas as pd

from src import data_prep
from src.svm_experiment import load_tier_splits
from src.tree_common import load_splits


def test_tree_split_matches_svm_split():
    svm = load_tier_splits(data_prep.clean(data_prep.load_raw()))
    tree = load_splits(False)
    assert tree["names"] == svm["names"]
    assert np.isclose(tree["threshold"], svm["threshold"])
    assert np.array_equal(tree["c_tr"], svm["y_tr"])
    assert np.array_equal(tree["c_va"], svm["y_va"])
    assert np.array_equal(tree["c_te"], svm["y_te"])
    assert np.allclose(tree["X_tr"], svm["X_tr_raw"])
    z_tr, z_va, z_te = data_prep.standardize(tree["X_tr"], tree["X_va"], tree["X_te"])
    assert np.allclose(z_va, svm["X_va"]) and np.allclose(z_te, svm["X_te"])
