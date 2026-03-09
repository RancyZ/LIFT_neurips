"""
Stage l2 — Generalizability: discriminative power on D_te.
Implements Section 3.2.4 of the LIFT paper.

All metrics are broken down by race group and stored in dicts.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from lift.schemas import StageResult


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy, specificity, sensitivity, FPR, and demographic parity for one group."""
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        spec, sens, fpr = 1.0, 0.0, 0.0
    dp = float(np.mean(y_pred == 1))
    return {"acc": acc, "spec": spec, "sens": sens, "fpr": fpr, "dp": dp}


def evaluate_generalizability(
    model_id: str,
    subsets: List[pd.DataFrame],
    D_te: pd.DataFrame,
    outcome_col: str,
    protected_col: str | None,
    best_params_per_b: List[dict],
    config: dict,
) -> StageResult:
    """
    For each b, for each race group g:
      Acc_b_g  = accuracy_score(y_test_race_g, y_pred_race_g)
      Spec_b_g = TN / (TN + FP)
      Sens_b_g = TP / (TP + FN)
      FPR_b_g  = FP / (FP + TN)
      DP_b_g   = mean(y_pred == 1)

    Acc_i_g, Spec_i_g, ... = mean over b for each group g.
    Acc_i, Spec_i = macro-average over groups (mean of group means).
    e_gen = alpha_gen[0]*Acc_i + alpha_gen[1]*Spec_i
    """
    from lift.models.model_factory import get_model

    alpha_gen = config.get("evaluation", {}).get("alpha_gen", [0.5, 0.5])

    drop_cols_te = [c for c in [outcome_col, protected_col] if c and c in D_te.columns]
    X_te = D_te.drop(columns=drop_cols_te).select_dtypes(include="number")
    y_te = D_te[outcome_col].values

    # Identify race groups from test set
    groups: List[Any] = []
    if protected_col and protected_col in D_te.columns:
        groups = sorted(D_te[protected_col].dropna().unique().tolist())

    # Accumulators: metric → group → list of per-b values
    metrics = ["acc", "spec", "sens", "fpr", "dp"]
    per_b: Dict[str, Dict[Any, List[float]]] = {
        m: {g: [] for g in (groups or ["overall"])} for m in metrics
    }

    for b_idx, subset in enumerate(subsets):
        if len(subset) < 2:
            continue

        drop_cols = [c for c in [outcome_col, protected_col] if c and c in subset.columns]
        X_b = subset.drop(columns=drop_cols).select_dtypes(include="number")
        y_b = subset[outcome_col]

        best_params = best_params_per_b[b_idx] if b_idx < len(best_params_per_b) else {}
        model = get_model(model_id, best_params)
        model.fit(X_b, y_b)

        X_te_aligned = X_te.reindex(columns=X_b.columns, fill_value=0)
        y_pred = model.predict(X_te_aligned)

        if groups:
            grp_col = D_te[protected_col].values
            for g in groups:
                mask = grp_col == g
                if mask.sum() == 0:
                    continue
                gm = _group_metrics(y_te[mask], y_pred[mask])
                for m in metrics:
                    per_b[m][g].append(gm[m])
        else:
            gm = _group_metrics(y_te, y_pred)
            for m in metrics:
                per_b[m]["overall"].append(gm[m])

    if not any(per_b["acc"][g] for g in per_b["acc"]):
        return StageResult(stage_id="l2_generalizability", model_id=model_id, score=0.0, raw={})

    # Average over b for each group
    group_means: Dict[str, Dict[Any, float]] = {m: {} for m in metrics}
    for m in metrics:
        for g in per_b[m]:
            vals = per_b[m][g]
            group_means[m][g] = float(np.mean(vals)) if vals else 0.0

    # Macro-average over groups
    Acc_i  = float(np.mean(list(group_means["acc"].values())))
    Spec_i = float(np.mean(list(group_means["spec"].values())))
    Sens_i = float(np.mean(list(group_means["sens"].values())))
    FPR_i  = float(np.mean(list(group_means["fpr"].values())))
    DP_i   = float(np.mean(list(group_means["dp"].values())))

    e_gen = alpha_gen[0] * Acc_i + alpha_gen[1] * Spec_i

    return StageResult(
        stage_id="l2_generalizability",
        model_id=model_id,
        score=float(e_gen),
        raw={
            "Acc_i":        Acc_i,
            "Spec_i":       Spec_i,
            "Sens_i":       Sens_i,
            "FPR_i":        FPR_i,
            "DP_i":         DP_i,
            "acc_by_group":  group_means["acc"],
            "spec_by_group": group_means["spec"],
            "sens_by_group": group_means["sens"],
            "fpr_by_group":  group_means["fpr"],
            "dp_by_group":   group_means["dp"],
        },
    )
