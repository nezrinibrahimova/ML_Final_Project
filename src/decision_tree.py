"""
decision_tree.py — decision tree FROM SCRATCH (NumPy only).

Supports:
  classification: criterion = "gini" | "entropy"   (Task B, price tier)
  regression:     criterion = "mse"                (Task A, price / log price)

Key ideas
---------
* Impurity:  Gini G = 1 - sum p_k^2 ; entropy H = -sum p_k log2 p_k ; MSE = variance.
* Split quality: impurity decrease
      dI = I(parent) - (N_L/N) I(left) - (N_R/N) I(right)
* Threshold search is VECTORISED: sort a feature once, then use cumulative
  class counts (or cumulative sum / sum of squares for regression) so every
  candidate threshold is scored in O(1). Total cost per node: O(d * n log n).
* Thresholds are midpoints between consecutive DISTINCT sorted values
  (same convention as scikit-learn).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node. Leaf if left and right are None."""

    def __init__(self):
        self.feature = None        # int, split feature (internal nodes)
        self.threshold = None      # float, go left if x[feature] <= threshold
        self.left = None
        self.right = None
        self.value = None          # leaf: class-probability vector OR mean
        self.n_samples = 0
        self.impurity = 0.0
        self.gain = 0.0            # weighted impurity decrease (for importances)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if task == "classification" and criterion not in ("gini", "entropy"):
            raise ValueError("classification needs criterion 'gini' or 'entropy'")
        if task == "regression" and criterion != "mse":
            raise ValueError("regression needs criterion 'mse'")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    # ------------------------------------------------------------------ impurity
    def _impurity_from_counts(self, counts):
        """counts: (..., K) class counts -> impurity (...). Vectorised."""
        n = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(n, 1)
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=-1)
        # entropy: define 0*log(0)=0
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log2(p), 0.0)
        return -np.sum(p * logp, axis=-1)

    def _impurity(self, y) -> float:
        """Impurity of a label vector. For classification y holds class
        indices 0..K-1 (encoded in fit); for regression it is the variance."""
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.var(y))
        counts = np.bincount(y, minlength=len(self.classes_)).astype(float)
        return float(self._impurity_from_counts(counts))

    # ------------------------------------------------------------------ split search
    def _best_split(self, X, y):
        """Return (feature, threshold, weighted_gain) or None.

        weighted_gain = (N/N_total) * dI  -> comparable to sklearn's
        min_impurity_decrease and used for feature importances.
        """
        n, d = X.shape
        parent_imp = self._impurity(y)
        msl = self.min_samples_leaf
        best = None
        best_dI = 0.0

        if self.task == "classification":
            K = len(self.classes_)
            onehot = np.eye(K)[y]                      # (n, K)

        for j in range(d):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            # candidate split i means: left = first i+1 samples
            valid = xs[:-1] != xs[1:]                   # only between distinct values
            if not valid.any():
                continue
            n_left = np.arange(1, n)                    # sizes of left child
            n_right = n - n_left
            valid &= (n_left >= msl) & (n_right >= msl)
            if not valid.any():
                continue

            if self.task == "classification":
                cl = np.cumsum(onehot[order], axis=0)[:-1]   # left counts (n-1, K)
                cr = onehot.sum(axis=0) - cl                  # right counts
                imp_l = self._impurity_from_counts(cl)
                imp_r = self._impurity_from_counts(cr)
            else:
                ys = y[order].astype(float)
                s = np.cumsum(ys)[:-1]
                s2 = np.cumsum(ys ** 2)[:-1]
                S, S2 = ys.sum(), (ys ** 2).sum()
                imp_l = s2 / n_left - (s / n_left) ** 2             # variance
                imp_r = (S2 - s2) / n_right - ((S - s) / n_right) ** 2
                imp_l = np.maximum(imp_l, 0.0)                      # numeric safety
                imp_r = np.maximum(imp_r, 0.0)

            dI = parent_imp - (n_left / n) * imp_l - (n_right / n) * imp_r
            dI = np.where(valid, dI, -np.inf)
            i = int(np.argmax(dI))
            if dI[i] > best_dI:                         # strict > : first best wins
                best_dI = float(dI[i])
                thr = (xs[i] + xs[i + 1]) / 2.0
                best = (j, thr, best_dI)

        if best is None:
            return None
        j, thr, dI = best
        return j, thr, dI * n / self._n_total

    # ------------------------------------------------------------------ grow
    def _leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        counts = np.bincount(y, minlength=len(self.classes_)).astype(float)
        return counts / counts.sum()                    # class probabilities

    def _grow(self, X, y, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        node.value = self._leaf_value(y)

        # ---- stopping rules ----
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or node.impurity <= 1e-12                    # pure node
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        j, thr, w_gain = split
        if w_gain < self.min_impurity_decrease:
            return node

        mask = X[:, j] <= thr
        node.feature, node.threshold, node.gain = j, thr, w_gain
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self._n_total = len(y)
        if self.task == "classification":
            self.classes_, y_enc = np.unique(y, return_inverse=True)
            y = y_enc
        else:
            y = y.astype(float)
        self.root = self._grow(X, y, depth=0)
        self._compute_importances()
        return self

    # ------------------------------------------------------------------ predict
    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _values(self, X):
        """Vectorised traversal: push index sets down the tree."""
        X = np.asarray(X, dtype=float)
        out = [None] * len(X)

        def rec(node, idx):
            if len(idx) == 0:
                return
            if node.is_leaf():
                for i in idx:
                    out[i] = node.value
                return
            m = X[idx, node.feature] <= node.threshold
            rec(node.left, idx[m])
            rec(node.right, idx[~m])

        rec(self.root, np.arange(len(X)))
        return out

    def predict(self, X):
        vals = self._values(X)
        if self.task == "regression":
            return np.array(vals, dtype=float)
        return self.classes_[np.argmax(np.vstack(vals), axis=1)]

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is for classification only")
        return np.vstack(self._values(X))

    # ------------------------------------------------------------------ extras
    def _compute_importances(self):
        imp = np.zeros(self.n_features_)

        def rec(node):
            if node.is_leaf():
                return
            imp[node.feature] += node.gain
            rec(node.left)
            rec(node.right)

        rec(self.root)
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp

    def get_depth(self) -> int:
        def rec(n):
            return 0 if n.is_leaf() else 1 + max(rec(n.left), rec(n.right))
        return rec(self.root)

    def get_n_leaves(self) -> int:
        def rec(n):
            return 1 if n.is_leaf() else rec(n.left) + rec(n.right)
        return rec(self.root)