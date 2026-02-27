"""
Fairness metric implementations.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np


def eop_score(tpr_dict: Dict[Any, float]) -> float:
    """Equal opportunity: 1 - max|TPR_a - TPR_a'| over all pairs."""
    vals = list(tpr_dict.values())
    if len(vals) < 2:
        return 1.0
    max_diff = max(abs(a - b) for a, b in combinations(vals, 2))
    return float(1.0 - max_diff)


def eq_odds_fpr_score(fpr_dict: Dict[Any, float]) -> float:
    """1 - max|FPR_a - FPR_a'|."""
    vals = list(fpr_dict.values())
    if len(vals) < 2:
        return 1.0
    max_diff = max(abs(a - b) for a, b in combinations(vals, 2))
    return float(1.0 - max_diff)


def demographic_parity_score(dp_dict: Dict[Any, float]) -> float:
    """1 - max|DP_a - DP_a'|."""
    vals = list(dp_dict.values())
    if len(vals) < 2:
        return 1.0
    max_diff = max(abs(a - b) for a, b in combinations(vals, 2))
    return float(1.0 - max_diff)


def aggregate_parity(
    tpr_dict: Dict[Any, float],
    fpr_dict: Dict[Any, float],
    dp_dict: Dict[Any, float],
    weights: List[float],
) -> float:
    """S_par = sum(alpha_par[k] * score_k)."""
    eop = eop_score(tpr_dict)
    fpr = eq_odds_fpr_score(fpr_dict)
    dp = demographic_parity_score(dp_dict)
    return float(weights[0] * eop + weights[1] * fpr + weights[2] * dp)
