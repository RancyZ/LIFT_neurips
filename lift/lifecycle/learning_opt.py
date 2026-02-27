"""
Stage l1 — Learning Optimisation: training time efficiency.
Implements Section 3.2.3 of the LIFT paper.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd

from lift.schemas import StageResult


def evaluate_learning_opt(
    model_id: str,
    subsets: List[pd.DataFrame],
    D_te: pd.DataFrame,
    outcome_col: str,
    config: dict,
) -> StageResult:
    """
    For each b in 1..B:
      1. Tune hyperparams via 5-fold CV within D_tr_b
      2. Retrain on full D_tr_b with best params
      3. Record training wall time T_i_b

    e_LeOp = mean(T_i_b for b in 1..B)  [lower = better]
    Normalised to [0,1] (inverted) for scoreboard.
    """
    from lift.models.model_factory import get_model
    from lift.models.tuner import tune_hyperparams

    hp_grid = config.get("models", {}).get("hyperparams", {}).get(model_id, {})
    cv_folds = config.get("pipeline", {}).get("cv_folds", 5)
    seed = config.get("pipeline", {}).get("random_seed", 42)

    times: List[float] = []

    for b_idx, subset in enumerate(subsets):
        if len(subset) < 2:
            continue

        X_b = subset.drop(columns=[outcome_col])
        y_b = subset[outcome_col]

        # Hyperparameter tuning
        best_params = tune_hyperparams(
            model_id=model_id,
            X_tr=X_b,
            y_tr=y_b,
            param_grid=hp_grid,
            cv_folds=min(cv_folds, len(subset) // 2),
            random_seed=seed + b_idx,
        )

        # Retrain and time
        model = get_model(model_id, best_params)
        t0 = time.perf_counter()
        model.fit(X_b, y_b)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    if not times:
        return StageResult(stage_id="l1_learning_opt", model_id=model_id, score=0.0, raw={})

    e_LeOp = float(np.mean(times))

    # Normalise: score is stored as raw time; pipeline normalises across models
    return StageResult(
        stage_id="l1_learning_opt",
        model_id=model_id,
        score=e_LeOp,          # will be inverted + normalised in runner
        raw={"mean_train_time_s": e_LeOp, "times_per_subset": times},
    )
