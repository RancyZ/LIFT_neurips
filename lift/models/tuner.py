"""
Hyperparameter tuning via k-fold cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Any, Dict


def tune_hyperparams(
    model_id: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    param_grid: Dict[str, list],
    cv_folds: int,
    random_seed: int,
) -> Dict[str, Any]:
    """
    5-fold CV hyperparameter search.
    Returns the best hyperparameter dict.
    """
    from lift.models.model_factory import get_model

    # Flatten param grid into list of param dicts
    param_list = _expand_grid(param_grid)

    best_params: Dict[str, Any] = param_list[0] if param_list else {}
    best_score = -1.0

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    for params in param_list:
        scores = []
        for fold_tr_idx, fold_val_idx in skf.split(X_tr, y_tr):
            X_fold_tr = X_tr.iloc[fold_tr_idx]
            y_fold_tr = y_tr.iloc[fold_tr_idx]
            X_fold_val = X_tr.iloc[fold_val_idx]
            y_fold_val = y_tr.iloc[fold_val_idx]

            model = get_model(model_id, params)
            try:
                model.fit(X_fold_tr, y_fold_tr)
                preds = model.predict(X_fold_val)
                scores.append(accuracy_score(y_fold_val, preds))
            except Exception:  # noqa: BLE001
                scores.append(0.0)

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params


def _expand_grid(param_grid: Dict[str, list]) -> list:
    """Cartesian product of all hyperparameter values."""
    import itertools
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
