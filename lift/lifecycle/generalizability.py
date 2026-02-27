"""
Stage l2 — Generalizability: discriminative power on D_te.
Implements Section 3.2.4 of the LIFT paper.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from lift.evaluation.metrics import accuracy, specificity
from lift.schemas import StageResult


def evaluate_generalizability(
    model_id: str,
    subsets: List[pd.DataFrame],
    D_te: pd.DataFrame,
    outcome_col: str,
    config: dict,
) -> StageResult:
    """
    For each b:
      Acc_b = accuracy_score(y_te, y_pred_b)
      Spec_b = TN_b / (TN_b + FP_b)

    Acc_i = mean(Acc_b)
    Spec_i = mean(Spec_b)
    e_gen = alpha_gen[0]*Acc_i + alpha_gen[1]*Spec_i
    """
    from lift.models.model_factory import get_model
    from lift.models.tuner import tune_hyperparams

    hp_grid = config.get("models", {}).get("hyperparams", {}).get(model_id, {})
    cv_folds = config.get("pipeline", {}).get("cv_folds", 5)
    seed = config.get("pipeline", {}).get("random_seed", 42)
    alpha_gen = config.get("evaluation", {}).get("alpha_gen", [0.5, 0.5])

    X_te = D_te.drop(columns=[outcome_col])
    y_te = D_te[outcome_col].values

    acc_per_b: List[float] = []
    spec_per_b: List[float] = []

    for b_idx, subset in enumerate(subsets):
        if len(subset) < 2:
            continue

        X_b = subset.drop(columns=[outcome_col])
        y_b = subset[outcome_col]

        best_params = tune_hyperparams(
            model_id=model_id,
            X_tr=X_b,
            y_tr=y_b,
            param_grid=hp_grid,
            cv_folds=min(cv_folds, len(subset) // 2),
            random_seed=seed + b_idx,
        )

        model = get_model(model_id, best_params)
        model.fit(X_b, y_b)

        # Align test columns
        X_te_aligned = X_te.reindex(columns=X_b.columns, fill_value=0)
        y_pred = model.predict(X_te_aligned)

        acc_per_b.append(accuracy(y_te, y_pred))
        spec_per_b.append(specificity(y_te, y_pred))

    if not acc_per_b:
        return StageResult(stage_id="l2_generalizability", model_id=model_id, score=0.0, raw={})

    Acc_i = float(np.mean(acc_per_b))
    Spec_i = float(np.mean(spec_per_b))
    e_gen = alpha_gen[0] * Acc_i + alpha_gen[1] * Spec_i

    return StageResult(
        stage_id="l2_generalizability",
        model_id=model_id,
        score=float(e_gen),
        raw={
            "Acc_i": Acc_i,
            "Spec_i": Spec_i,
            "Acc_per_b": acc_per_b,
            "Spec_per_b": spec_per_b,
        },
    )
