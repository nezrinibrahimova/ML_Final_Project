"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def _as_float(y_true, y_pred):
    """Convert to flat float arrays and validate inputs."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if y_true.size == 0:
        raise ValueError("input arrays cannot be empty")

    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("input arrays must contain only finite values")

    return y_true, y_pred

def rmse(y_true, y_pred) -> float:
    """Root mean squared error: sqrt(mean((y - y_hat)^2))."""
    y_true, y_pred = _as_float(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error: mean(|y - y_hat|)."""
    y_true, y_pred = _as_float(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y_true, y_pred = _as_float(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def _as_labels(y_true, y_pred):
    """Validate binary labels and convert to flat integer arrays."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if y_true.size == 0:
        raise ValueError("input arrays cannot be empty")

    if not (np.isin(y_true, [0, 1]).all()
            and np.isin(y_pred, [0, 1]).all()):
        raise ValueError("labels must be 0 or 1")

    return y_true.astype(int), y_pred.astype(int)


def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _as_labels(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def _average_ranks(x):
    """Ranks starting at 1; tied values get the average of their ranks."""
    x = np.asarray(x, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    first = last - counts + 1
    return ((first + last) / 2.0)[inverse]


def roc_auc(y_true, scores) -> float:
    """Compute ROC-AUC using average ranks (Mann-Whitney)."""
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {scores.shape}")

    if y_true.size == 0:
        raise ValueError("input arrays cannot be empty")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")

    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    y_true = y_true.astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes in y_true")

    ranks = _average_ranks(scores)
    r_pos = ranks[y_true == 1].sum()

    return float(
        (r_pos - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


# --- Comparison helper ------------------------------------------------------
def _fmt(v) -> str:
    """Format a value for the table: big numbers with commas, small with 4 decimals."""
    if v is None:
        return "-"
    if isinstance(v, (int, float, np.floating, np.integer)):
        v = float(v)
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4f}"
    return str(v)


def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.

    name_to_metrics = {"model name": {"RMSE": 1.0, "R2": 0.5}, ...}
    """
    metric_names = []
    for metrics in name_to_metrics.values():
        for k in metrics:
            if k not in metric_names:
                metric_names.append(k)

    rows = [[name] + [_fmt(m.get(k)) for k in metric_names]
            for name, m in name_to_metrics.items()]
    header = ["model"] + metric_names
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                         for i, c in enumerate(cells))

    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))