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
    protected_col: str | None,
    best_params_per_b: List[dict],
    config: dict,
) -> StageResult:
    """
    For each b in 1..B:
      1. Train on D_tr_b using pre-tuned hyperparams (tuning done once in runner)
      2. Record training wall time T_i_b

    e_LeOp = mean(T_i_b for b in 1..B)  [lower = better]
    Normalised to [0,1] (inverted) for scoreboard.
    """
    from lift.models.model_factory import get_model

    times: List[float] = []

    for b_idx, subset in enumerate(subsets):
        if len(subset) < 2:
            continue

        drop_cols = [c for c in [outcome_col, protected_col] if c and c in subset.columns]
        X_b = subset.drop(columns=drop_cols).select_dtypes(include="number")
        y_b = subset[outcome_col]

        best_params = best_params_per_b[b_idx] if b_idx < len(best_params_per_b) else {}

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
