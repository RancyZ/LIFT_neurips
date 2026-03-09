"""
Stage l4 — Monitoring: Wasserstein drift between D_tr subsets and D_te.
Implements Section 3.2.6 of the LIFT paper.

NOTE: e_MoUp is data-driven, NOT model-dependent.
All models get the same e_MoUp for a given dataset run.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from lift.evaluation.metrics import wasserstein_mean
from lift.schemas import StageResult


def evaluate_monitoring(
    subsets: List[pd.DataFrame],
    D_te: pd.DataFrame,
    outcome_col: str,
    protected_col: str | None,
    config: dict,
    model_id: str = "dataset",
) -> StageResult:
    """
    For each b:
      W_b = (1/P) * sum_p W1(F_tr_p_b, F_te_p)
    e_MoUp = mean(W_b)  [higher = more drift = higher risk]

    Score stored as raw drift value; pipeline inverts for scoreboard.
    """
    drop_cols_te = [c for c in [outcome_col, protected_col] if c and c in D_te.columns]
    X_te = D_te.drop(columns=drop_cols_te).select_dtypes(include="number")

    w_per_b: List[float] = []
    for subset in subsets:
        drop_cols = [c for c in [outcome_col, protected_col] if c and c in subset.columns]
        X_b = subset.drop(columns=drop_cols).select_dtypes(include="number")
        w = wasserstein_mean(X_b, X_te)
        w_per_b.append(w)

    if not w_per_b:
        return StageResult(stage_id="l4_monitoring", model_id=model_id, score=0.0, raw={})

    e_MoUp = float(np.mean(w_per_b))

    # Score = raw drift (pipeline normalises + inverts)
    return StageResult(
        stage_id="l4_monitoring",
        model_id=model_id,
        score=e_MoUp,   # will be inverted + normalised in runner
        raw={"e_MoUp": e_MoUp, "W_per_b": w_per_b},
    )
