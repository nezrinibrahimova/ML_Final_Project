"""Ridge regression from scratch using NumPy."""

import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X and y must contain only finite values")

        x_mean = X.mean(axis=0)
        y_mean = y.mean()

        X_centered = X - x_mean
        y_centered = y - y_mean

        # Minimize: sum((y - prediction)**2) + alpha * sum(coef**2).
        # The intercept is not penalized.
        matrix = (
            X_centered.T @ X_centered
            + self.alpha * np.eye(X.shape[1])
        )
        rhs = X_centered.T @ y_centered

        self.coef_ = np.linalg.solve(matrix, rhs)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != len(self.coef_):
            raise ValueError("X has an unexpected number of features")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        return X @ self.coef_ + self.intercept_