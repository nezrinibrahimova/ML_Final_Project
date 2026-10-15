"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

Pegasos minimises the regularised hinge-loss primal by stochastic
sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. For a mini-batch A_t of size k and the set
A_t+ = {i in A_t : y_i (w . x_i + b) < 1} of hinge-active points:

    w <- (1 - eta_t*lambda) w + (eta_t/k) * sum_{i in A_t+} y_i x_i
    b <- b                    + (eta_t/k) * sum_{i in A_t+} y_i

The bias is NOT regularised (it is not in ||w||^2), so it gets its own
plain sub-gradient step. k = 1 gives the classic single-sample Pegasos.

Learning-rate schedules (``lr_schedule``):
  - "pegasos":  eta_t = 1 / (lambda * (t + t0))
        t0 = 0 is the paper's schedule. Its first step is eta_1 = 1/lambda
        (1e4 for lambda = 1e-4); w is shrunk back by the (1 - eta*lambda)
        factor, but the UNREGULARISED bias is not, so one early violator can
        throw b to +/-1e3 and it takes many epochs to come back. Default
        t0 = 1/lambda caps the first step at eta_1 ~= 1 and fixes this.
  - "invsqrt":  eta_t = eta0 / sqrt(t)
  - "constant": eta_t = eta0

Non-linear extension: RANDOM FOURIER FEATURES (Rahimi & Recht, 2007).
``RandomFourierFeatures`` maps X -> phi(X) so that
phi(x) . phi(x') ~= exp(-gamma ||x - x'||^2) (RBF kernel); LINEAR Pegasos
on phi(X) then approximates an RBF-kernel SVM. We chose it over kernelised
Pegasos because training stays O(n * D) per epoch instead of O(n^2) kernel
evaluations, which matters on 100k+ listings, and prediction does not need
to keep the support vectors around.

scikit-learn's SVC is a BASELINE only — it is not used anywhere here.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


def hinge_objective(w, b, X, y_pm, lambda_) -> float:
    """J(w, b) on (X, y_pm) with y_pm in {-1, +1}."""
    margins = y_pm * (X @ w + b)
    return float(0.5 * lambda_ * (w @ w) + np.maximum(0.0, 1.0 - margins).mean())


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength (~ 1/(n*C))
        n_iters: int = 20_000,       # number of sub-gradient steps
        n_epochs: int | None = None, # if set, overrides n_iters (passes over data)
        batch_size: int = 1,         # k in the update above
        lr_schedule: str = "pegasos",  # "pegasos" | "invsqrt" | "constant"
        eta0: float = 0.1,           # base rate for "invsqrt" / "constant"
        t0: float | None = None,     # offset for "pegasos"; None -> 1/lambda
        fit_intercept: bool = True,
        project: bool = False,       # project w onto the ball ||w|| <= 1/sqrt(lambda)
        track_every: int | None = None,  # steps between J(w,b) evaluations
        sv_tol: float = 1e-6,        # y f(x) <= 1 + sv_tol  => support vector
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.t0 = t0
        self.fit_intercept = fit_intercept
        self.project = project
        self.track_every = track_every
        self.sv_tol = sv_tol
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias

    # --- labels -------------------------------------------------------------
    def _to_pm1(self, y):
        """Map any 2-class labels to {-1, +1}; classes_[1] becomes +1."""
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"PegasosSVM is binary; got {classes.size} classes")
        self.classes_ = classes
        return np.where(y == classes[1], 1.0, -1.0)

    # --- training -----------------------------------------------------------
    def _eta(self, t: int) -> float:
        if self.lr_schedule == "pegasos":
            t0 = 1.0 / self.lambda_ if self.t0 is None else self.t0
            return 1.0 / (self.lambda_ * (t + t0))
        if self.lr_schedule == "invsqrt":
            return self.eta0 / np.sqrt(t)
        if self.lr_schedule == "constant":
            return self.eta0
        raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")

    def _batches(self, n: int, rng):
        """Yield index arrays: shuffled epochs if n_epochs, else i.i.d. sampling."""
        k = self.batch_size
        if self.n_epochs is not None:
            for _ in range(self.n_epochs):
                perm = rng.permutation(n)
                for start in range(0, n, k):
                    yield perm[start:start + k]
        else:
            for _ in range(self.n_iters):
                yield rng.integers(0, n, size=k)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_pm = self._to_pm1(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        total = (self.n_epochs * int(np.ceil(n / self.batch_size))
                 if self.n_epochs is not None else self.n_iters)
        track_every = self.track_every or max(1, total // 200)
        radius = 1.0 / np.sqrt(self.lambda_)

        w = np.zeros(d)
        b = 0.0
        self.obj_history_ = [hinge_objective(w, b, X, y_pm, self.lambda_)]
        self.obj_steps_ = [0]

        for t, idx in enumerate(self._batches(n, rng), start=1):
            eta = self._eta(t)
            Xb, yb = X[idx], y_pm[idx]
            active = yb * (Xb @ w + b) < 1.0
            w *= 1.0 - eta * self.lambda_
            if active.any():
                scale = eta / idx.size
                w += scale * (yb[active] @ Xb[active])
                if self.fit_intercept:
                    b += scale * yb[active].sum()
            if self.project:
                norm = np.linalg.norm(w)
                if norm > radius:
                    w *= radius / norm
            if t % track_every == 0 or t == total:
                self.obj_history_.append(hinge_objective(w, b, X, y_pm, self.lambda_))
                self.obj_steps_.append(t)

        self.w, self.b = w, float(b)
        self.n_steps_ = total

        # --- geometry of the solution -------------------------------------
        self.w_norm_ = float(np.linalg.norm(w))
        self.margin_ = 2.0 / self.w_norm_ if self.w_norm_ > 0 else np.inf
        functional = y_pm * self.decision_function(X)
        self.support_ = np.flatnonzero(functional <= 1.0 + self.sv_tol)
        self.n_support_ = int(self.support_.size)
        self.objective_ = self.obj_history_[-1]
        return self

    # --- inference ----------------------------------------------------------
    def decision_function(self, X):
        """Return the signed score  w . x + b  for each row."""
        if self.w is None:
            raise RuntimeError("PegasosSVM is not fitted yet")
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Map sign(w . x + b) back to the original labels (ties -> classes_[1])."""
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class RandomFourierFeatures:
    """
    phi(x) = sqrt(2/D) * cos(W^T x + c),  W ~ N(0, 2*gamma*I),  c ~ U[0, 2*pi]
    so that  E[phi(x) . phi(x')] = exp(-gamma * ||x - x'||^2).
    Only the input dimension is learned in fit(); W and c are pure randomness.
    """

    def __init__(self, n_components: int = 500, gamma: float = 1.0, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        d = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=float) @ self.W_ + self.c_
        return np.sqrt(2.0 / self.n_components) * np.cos(Z)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class RFFPegasosSVM:
    """Approximate RBF-kernel SVM: RandomFourierFeatures -> linear PegasosSVM."""

    def __init__(self, n_components: int = 500, gamma: float = 1.0,
                 random_state: int = 42, **svm_params):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.svm_params = svm_params

    def fit(self, X, y):
        self.rff_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state)
        Z = self.rff_.fit_transform(X)
        self.svm_ = PegasosSVM(random_state=self.random_state, **self.svm_params).fit(Z, y)
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.rff_.transform(X))

    def predict(self, X):
        return self.svm_.predict(self.rff_.transform(X))

    def __getattr__(self, name):
        # expose obj_history_, margin_, n_support_, w, b, ... of the inner SVM
        if name.startswith("__") or "svm_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["svm_"], name)
