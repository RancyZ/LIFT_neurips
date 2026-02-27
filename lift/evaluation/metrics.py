"""
Core metric implementations.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import wasserstein_distance


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        denom = tn + fp
        return float(tn / denom) if denom > 0 else 1.0
    # Edge case: only one class
    return 1.0


def tpr_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[Any, float]:
    """Maps group value → TPR within that group."""
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        denom = tp + fn
        result[g] = float(tp / denom) if denom > 0 else 0.0
    return result


def fpr_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[Any, float]:
    """Maps group value → FPR within that group."""
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        denom = fp + tn
        result[g] = float(fp / denom) if denom > 0 else 0.0
    return result


def positive_rate_per_group(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[Any, float]:
    """Demographic parity rate per group."""
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        yp = y_pred[mask]
        result[g] = float(np.mean(yp == 1)) if len(yp) > 0 else 0.0
    return result


def wasserstein_mean(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> float:
    """
    Feature-averaged 1-Wasserstein distance.
    scipy.stats.wasserstein_distance per column, then mean.
    """
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    if not common_cols:
        return 0.0
    distances = [
        wasserstein_distance(X_tr[c].dropna().values, X_te[c].dropna().values)
        for c in common_cols
    ]
    return float(np.mean(distances))
