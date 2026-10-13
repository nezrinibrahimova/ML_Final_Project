"""Run from repo root:  python -m pytest tests/test_tree.py -q   (or python tests/test_tree.py)"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.decision_tree import DecisionTree


def _tiny_cls():
    X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_root_split_matches_sklearn_tiny():
    X, y = _tiny_cls()
    for crit in ("gini", "entropy"):
        mine = DecisionTree("classification", crit).fit(X, y)
        sk = DecisionTreeClassifier(criterion=crit, random_state=0).fit(X, y)
        assert mine.root.feature == sk.tree_.feature[0]
        assert np.isclose(mine.root.threshold, sk.tree_.threshold[0])
        assert (mine.predict(X) == sk.predict(X)).all()


def test_regression_tiny_matches_sklearn():
    X = np.array([[1.], [2.], [3.], [4.], [5.], [6.]])
    y = np.array([1., 1.2, 0.9, 5., 5.1, 4.9])
    mine = DecisionTree("regression", "mse", max_depth=2).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    assert np.isclose(mine.root.threshold, sk.tree_.threshold[0])
    assert np.allclose(mine.predict(X), sk.predict(X))


def test_full_tree_same_predictions_random_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mine = DecisionTree("classification", "gini", max_depth=4).fit(X, y)
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    assert (mine.predict(X) == sk.predict(X)).mean() > 0.97   # ties may differ


def test_predict_proba_sums_to_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3)); y = rng.integers(0, 2, 100)
    p = DecisionTree("classification", "gini", max_depth=3).fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_each_hyperparameter_changes_the_tree():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 4)); y = X[:, 0] * 2 + rng.normal(size=400)
    base = DecisionTree("regression", "mse").fit(X, y)
    assert DecisionTree("regression", "mse", max_depth=2).fit(X, y).get_depth() <= 2
    assert DecisionTree("regression", "mse", min_samples_split=100).fit(X, y).get_n_leaves() < base.get_n_leaves()
    t = DecisionTree("regression", "mse", min_samples_leaf=30).fit(X, y)
    def leaf_sizes(n): return [n.n_samples] if n.is_leaf() else leaf_sizes(n.left) + leaf_sizes(n.right)
    assert min(leaf_sizes(t.root)) >= 30
    assert DecisionTree("regression", "mse", min_impurity_decrease=0.1).fit(X, y).get_n_leaves() < base.get_n_leaves()


def test_feature_importances():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4)); y = (X[:, 2] > 0).astype(int)
    imp = DecisionTree("classification", "gini", max_depth=3).fit(X, y).feature_importances_
    assert np.isclose(imp.sum(), 1.0) and imp.argmax() == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)