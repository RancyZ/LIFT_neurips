"""
SHAP-based explanation consistency metrics.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Set

import numpy as np
import pandas as pd


def compute_shap_top_H(
    model,
    X: pd.DataFrame,
    groups: pd.Series,
    H: int,
) -> Dict[Any, Set[str]]:
    """
    Computes SHAP values for X, then for each group a ∈ A
    returns the set of H features with highest mean |SHAP|.
    Uses shap.Explainer (model-agnostic).
    """
    import shap

    # Choose fastest explainer for known model types
    model_type = type(model).__name__
    try:
        if model_type in ("RandomForestWrapper", "DecisionTreeWrapper"):
            explainer = shap.TreeExplainer(model._model)
            shap_values = explainer.shap_values(X)
            # For binary classification, shap_values may be a list [neg, pos]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif model_type in ("LogisticRegressionWrapper",):
            explainer = shap.LinearExplainer(model._model, X)
            shap_values = explainer.shap_values(X)
        else:
            # Fallback: KernelExplainer with a small background sample
            background = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
                background,
            )
            shap_values = explainer.shap_values(X, nsamples=100)
    except Exception:  # noqa: BLE001
        # Return empty sets on failure — S_exp will be 0
        return {g: set() for g in groups.unique()}

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    top_H_sets: Dict[Any, Set[str]] = {}
    for g in groups.unique():
        mask = (groups == g).values
        if mask.sum() == 0:
            top_H_sets[g] = set()
            continue
        mean_abs = shap_df.iloc[mask].abs().mean()
        top_features = set(mean_abs.nlargest(H).index.tolist())
        top_H_sets[g] = top_features

    return top_H_sets


def explanation_consistency(
    top_H_sets: Dict[Any, Set[str]],
    H: int,
) -> float:
    """
    S_exp = avg pairwise |H_a ∩ H_a'| / H over all pairs (a, a').
    Returns float ∈ [0, 1].
    """
    group_list = list(top_H_sets.keys())
    if len(group_list) < 2 or H == 0:
        return 1.0

    overlaps = []
    for a, b in combinations(group_list, 2):
        intersection = len(top_H_sets[a] & top_H_sets[b])
        overlaps.append(intersection / H)

    return float(np.mean(overlaps))
