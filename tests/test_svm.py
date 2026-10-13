"""Run from repo root:  python -m pytest tests/test_svm.py -q"""
import numpy as np
from sklearn.datasets import make_circles, make_classification
from sklearn.svm import LinearSVC
from src.svm import PegasosSVM, RFFPegasosSVM, RandomFourierFeatures, hinge_objective


def _separable(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2, 0.5, (n // 2, 2)), rng.normal(2, 0.5, (n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def _noisy(n=3000):
    X, y = make_classification(n, 10, n_informative=5, flip_y=0.05, random_state=0)
    return (X - X.mean(0)) / X.std(0), y


def test_separable_data_is_classified_perfectly():
    X, y = _separable()
    m = PegasosSVM(lambda_=1e-2, n_epochs=10, batch_size=8).fit(X, y)
    assert (m.predict(X) == y).all()


def test_objective_decreases():
    X, y = _noisy()
    m = PegasosSVM(lambda_=1e-3, n_epochs=10, batch_size=32).fit(X, y)
    J = np.array(m.obj_history_)
    assert J[0] == 1.0                      # w = 0, b = 0  =>  hinge = 1 everywhere
    assert J[-1] < 0.5 * J[0]
    assert J[-5:].mean() < J[len(J) // 4]   # still going down in the second half


def test_objective_close_to_sklearn_optimum():
    X, y = _noisy()
    lam = 1e-3
    y_pm = np.where(y == 1, 1.0, -1.0)
    m = PegasosSVM(lambda_=lam, n_epochs=30, batch_size=32).fit(X, y)
    sk = LinearSVC(C=1 / (lam * len(y)), loss="hinge", max_iter=100_000, tol=1e-8).fit(X, y)
    J_sk = hinge_objective(sk.coef_[0], sk.intercept_[0], X, y_pm, lam)
    assert m.objective_ <= 1.02 * J_sk
    assert (m.predict(X) == sk.predict(X)).mean() > 0.97


def test_arbitrary_labels_round_trip():
    X, y = _separable()
    labels = np.where(y == 1, "premium", "standard")
    m = PegasosSVM(lambda_=1e-2, n_epochs=5).fit(X, labels)
    assert set(m.predict(X)) == {"premium", "standard"}
    assert (m.predict(X) == labels).all()
    # classes_[1] ("standard", sorted last) is the +1 class
    assert (m.decision_function(X)[labels == "standard"] > 0).all()


def test_margin_and_support_vectors():
    X, y = _noisy()
    m = PegasosSVM(lambda_=1e-3, n_epochs=5, batch_size=16).fit(X, y)
    assert np.isclose(m.margin_, 2 / np.linalg.norm(m.w))
    y_pm = np.where(y == 1, 1.0, -1.0)
    func = y_pm * m.decision_function(X)
    assert set(m.support_) == set(np.flatnonzero(func <= 1 + m.sv_tol))
    assert 0 < m.n_support_ < len(y)


def test_stronger_regularisation_gives_wider_margin():
    X, y = _noisy()
    small = PegasosSVM(lambda_=1e-4, n_epochs=10, batch_size=32).fit(X, y)
    big = PegasosSVM(lambda_=1e-1, n_epochs=10, batch_size=32).fit(X, y)
    assert big.margin_ > small.margin_
    assert big.n_support_ > small.n_support_


def test_all_lr_schedules_learn():
    X, y = _noisy()
    for kw in ({"lr_schedule": "pegasos"}, {"lr_schedule": "pegasos", "t0": 0},
               {"lr_schedule": "invsqrt", "eta0": 0.5}, {"lr_schedule": "constant", "eta0": 0.01}):
        m = PegasosSVM(lambda_=1e-3, n_epochs=5, batch_size=32, **kw).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.8, kw


def test_default_offset_keeps_bias_stable():
    # t0 = 0 (paper) lets the unregularised bias blow up on separable data
    X, y = _separable()
    paper = PegasosSVM(lambda_=1e-3, n_epochs=10, t0=0).fit(X, y)
    ours = PegasosSVM(lambda_=1e-3, n_epochs=10).fit(X, y)
    assert abs(paper.b) > 100 and abs(ours.b) < 5
    assert ours.objective_ < paper.objective_


def test_seed_makes_training_deterministic():
    X, y = _noisy(1000)
    a = PegasosSVM(lambda_=1e-3, n_iters=2000, random_state=7).fit(X, y)
    b = PegasosSVM(lambda_=1e-3, n_iters=2000, random_state=7).fit(X, y)
    c = PegasosSVM(lambda_=1e-3, n_iters=2000, random_state=8).fit(X, y)
    assert np.array_equal(a.w, b.w) and a.b == b.b
    assert not np.array_equal(a.w, c.w)


def test_rff_approximates_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    gamma = 0.5
    Z = RandomFourierFeatures(n_components=20_000, gamma=gamma).fit_transform(X)
    K = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    assert np.abs(Z @ Z.T - K).max() < 0.05


def test_rff_solves_nonlinear_problem_linear_cannot():
    X, y = make_circles(1500, noise=0.1, factor=0.4, random_state=0)
    lin = PegasosSVM(lambda_=1e-3, n_epochs=20, batch_size=32).fit(X, y)
    rff = RFFPegasosSVM(n_components=300, gamma=1.0, lambda_=1e-3, n_epochs=20, batch_size=32).fit(X, y)
    assert (lin.predict(X) == y).mean() < 0.7
    assert (rff.predict(X) == y).mean() > 0.95
    assert rff.margin_ > 0 and len(rff.obj_history_) > 1


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
