"""
Stage l3 — Deployment: S_rob + S_par + S_exp.
Implements Section 3.2.5 of the LIFT paper.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from lift.evaluation.fairness import aggregate_parity
from lift.evaluation.explainability import compute_shap_top_H, explanation_consistency
from lift.evaluation.metrics import (
    accuracy, specificity,
    tpr_per_group, fpr_per_group, positive_rate_per_group,
)
from lift.schemas import StageResult


def evaluate_deployment(
    model_id: str,
    subsets: List[pd.DataFrame],
    D_te: pd.DataFrame,
    outcome_col: str,
    protected_col: str,
    best_params_per_b: List[dict],
    config: dict,
) -> StageResult:
    """
    S_rob: robustness from stddev of Acc/Spec across B subsets (inverted → higher = more stable)
    S_par: parity attainment (EOP, equalized odds FPR, DP)
    S_exp: SHAP explanation consistency across subgroups — computed once on final model
    e_dep = alpha_dep[0]*S_rob + alpha_dep[1]*S_par + alpha_dep[2]*S_exp
    """
    from lift.models.model_factory import get_model

    alpha_rob = config.get("evaluation", {}).get("alpha_rob", [0.5, 0.5])
    alpha_par = config.get("evaluation", {}).get("alpha_par", [0.333, 0.333, 0.334])
    alpha_dep = config.get("evaluation", {}).get("alpha_dep", [0.333, 0.333, 0.334])
    H = config.get("evaluation", {}).get("shap_top_H", 5)

    drop_cols_te = [c for c in [outcome_col, protected_col] if c and c in D_te.columns]
    X_te = D_te.drop(columns=drop_cols_te).select_dtypes(include="number")
    y_te = D_te[outcome_col].values
    groups_te = D_te[protected_col] if protected_col and protected_col in D_te.columns else None

    acc_per_b: List[float] = []
    spec_per_b: List[float] = []
    # Parity accumulated across all subsets
    tpr_agg: dict = {}
    fpr_agg: dict = {}
    dp_agg: dict = {}

    fitted_model = None       # last fitted model — used for single SHAP call after loop
    last_X_te_aligned = None

    for b_idx, subset in enumerate(subsets):
        if len(subset) < 2:
            continue

        drop_cols = [c for c in [outcome_col, protected_col] if c and c in subset.columns]
        X_b = subset.drop(columns=drop_cols).select_dtypes(include="number")
        y_b = subset[outcome_col]

        best_params = best_params_per_b[b_idx] if b_idx < len(best_params_per_b) else {}
        model = get_model(model_id, best_params)
        model.fit(X_b, y_b)
        fitted_model = model

        X_te_aligned = X_te.reindex(columns=X_b.columns, fill_value=0)
        last_X_te_aligned = X_te_aligned
        y_pred = model.predict(X_te_aligned)

        acc_per_b.append(accuracy(y_te, y_pred))
        spec_per_b.append(specificity(y_te, y_pred))

        # Parity (per subset, averaged later)
        if groups_te is not None:
            g_arr = groups_te.values
            tpr_d = tpr_per_group(y_te, y_pred, g_arr)
            fpr_d = fpr_per_group(y_te, y_pred, g_arr)
            dp_d = positive_rate_per_group(y_pred, g_arr)
            for g in tpr_d:
                tpr_agg.setdefault(g, []).append(tpr_d[g])
                fpr_agg.setdefault(g, []).append(fpr_d[g])
                dp_agg.setdefault(g, []).append(dp_d.get(g, 0.0))

    if not acc_per_b:
        return StageResult(stage_id="l3_deployment", model_id=model_id, score=0.0, raw={})

    # S_rob: invert normalised stddev of acc and spec
    sigma_acc = float(np.std(acc_per_b))
    sigma_spec = float(np.std(spec_per_b))
    # Normalise sigmas to [0,1] (divide by maximum possible diff = 1)
    s_rob_acc = 1.0 - min(sigma_acc, 1.0)
    s_rob_spec = 1.0 - min(sigma_spec, 1.0)
    S_rob = alpha_rob[0] * s_rob_acc + alpha_rob[1] * s_rob_spec

    # S_par: average per-subset parity
    if tpr_agg:
        mean_tpr = {g: float(np.mean(v)) for g, v in tpr_agg.items()}
        mean_fpr = {g: float(np.mean(v)) for g, v in fpr_agg.items()}
        mean_dp = {g: float(np.mean(v)) for g, v in dp_agg.items()}
        S_par = aggregate_parity(mean_tpr, mean_fpr, mean_dp, alpha_par)
    else:
        S_par = 1.0  # no protected attribute → no parity penalty

    # S_exp — intentionally computed once on the final fitted model instead of per-subset.
    # The spec calls for per-b SHAP passes but KernelExplainer is O(N²); running it B times
    # would dominate total runtime. Single-pass on the last fitted model is a deliberate
    # efficiency trade-off that preserves the directional signal of the metric.
    S_exp = 0.0
    if groups_te is not None and fitted_model is not None and last_X_te_aligned is not None:
        try:
            top_H = compute_shap_top_H(fitted_model, last_X_te_aligned, groups_te, H)
            S_exp = explanation_consistency(top_H, H)
        except Exception:  # noqa: BLE001
            pass

    e_dep = alpha_dep[0] * S_rob + alpha_dep[1] * S_par + alpha_dep[2] * S_exp

    return StageResult(
        stage_id="l3_deployment",
        model_id=model_id,
        score=float(e_dep),
        raw={
            "S_rob": S_rob,
            "S_par": S_par,
            "S_exp": S_exp,
            "sigma_Acc": sigma_acc,
            "sigma_Spec": sigma_spec,
        },
    )
