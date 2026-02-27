"""Unit tests for evaluation/metrics.py and evaluation/fairness.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.evaluation.metrics import (
    accuracy, specificity, tpr_per_group, fpr_per_group,
    positive_rate_per_group, wasserstein_mean,
)
from lift.evaluation.fairness import eop_score, eq_odds_fpr_score, demographic_parity_score
import pandas as pd


# ---------------------------------------------------------------------------
# test_specificity
# ---------------------------------------------------------------------------

def test_specificity_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert specificity(y_true, y_pred) == 1.0


def test_specificity_all_negative_predictions():
    """All predicted 0: TN=2, FP=0 → spec=1.0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    assert specificity(y_true, y_pred) == 1.0


def test_specificity_all_positive_predictions():
    """All predicted 1: TN=0, FP=2 → spec=0.0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    assert specificity(y_true, y_pred) == 0.0


def test_specificity_random():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    # TN=2, FP=1 → spec = 2/3
    assert abs(specificity(y_true, y_pred) - 2/3) < 1e-6


# ---------------------------------------------------------------------------
# test_eop_score
# ---------------------------------------------------------------------------

def test_eop_score_equal_tpr():
    """EOP = 1 when TPRs equal."""
    tpr = {"A": 0.8, "B": 0.8}
    assert eop_score(tpr) == 1.0


def test_eop_score_max_gap():
    """EOP = 0 when TPR difference is 1."""
    tpr = {"A": 0.0, "B": 1.0}
    assert eop_score(tpr) == 0.0


def test_eop_score_partial():
    tpr = {"A": 0.6, "B": 0.8}
    assert abs(eop_score(tpr) - 0.8) < 1e-6


def test_eop_score_single_group():
    """Single group → no pairs → score = 1."""
    assert eop_score({"A": 0.5}) == 1.0


# ---------------------------------------------------------------------------
# test_wasserstein
# ---------------------------------------------------------------------------

def test_wasserstein_identical_distributions():
    """W1 = 0 when distributions are identical."""
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    result = wasserstein_mean(df, df)
    assert result == 0.0


def test_wasserstein_disjoint():
    """W1 > 0 when distributions are separated."""
    df_tr = pd.DataFrame({"f1": [0.0, 0.0, 0.0]})
    df_te = pd.DataFrame({"f1": [10.0, 10.0, 10.0]})
    result = wasserstein_mean(df_tr, df_te)
    assert result == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# test_demographic_parity
# ---------------------------------------------------------------------------

def test_dp_equal_rates():
    dp = {"A": 0.5, "B": 0.5}
    assert demographic_parity_score(dp) == 1.0


def test_dp_max_disparity():
    dp = {"A": 0.0, "B": 1.0}
    assert demographic_parity_score(dp) == 0.0
