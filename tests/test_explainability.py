"""Unit tests for evaluation/explainability.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.evaluation.explainability import explanation_consistency


# ---------------------------------------------------------------------------
# test_shap_overlap
# ---------------------------------------------------------------------------

def test_explanation_consistency_identical():
    """S_exp = 1 when sets are identical."""
    sets = {"A": {"f1", "f2", "f3"}, "B": {"f1", "f2", "f3"}}
    result = explanation_consistency(sets, H=3)
    assert result == 1.0


def test_explanation_consistency_disjoint():
    """S_exp = 0 when sets are completely disjoint."""
    sets = {"A": {"f1", "f2"}, "B": {"f3", "f4"}}
    result = explanation_consistency(sets, H=2)
    assert result == 0.0


def test_explanation_consistency_partial():
    """S_exp = 0.5 when half of features overlap."""
    sets = {"A": {"f1", "f2"}, "B": {"f1", "f3"}}
    result = explanation_consistency(sets, H=2)
    assert result == pytest.approx(0.5)


def test_explanation_consistency_single_group():
    """Single group → no pairs → score = 1.0."""
    sets = {"A": {"f1", "f2"}}
    result = explanation_consistency(sets, H=2)
    assert result == 1.0


def test_explanation_consistency_h_zero():
    """H=0 edge case → returns 1.0."""
    sets = {"A": set(), "B": set()}
    result = explanation_consistency(sets, H=0)
    assert result == 1.0


def test_explanation_consistency_three_groups():
    """Three groups — average of 3 pairs."""
    # pairs: (A,B)=1/2, (A,C)=1/2, (B,C)=2/2
    sets = {
        "A": {"f1", "f2"},
        "B": {"f1", "f3"},
        "C": {"f1", "f3"},
    }
    result = explanation_consistency(sets, H=2)
    # (1 + 1 + 2) / (2 * 3) = 4/6 ≈ 0.6667
    assert abs(result - (1 + 1 + 2) / (3 * 2)) < 1e-6
