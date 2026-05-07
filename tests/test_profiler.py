"""Unit tests for DataProfiler — deterministic stat computations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.agents.data_profiler import DataProfiler
from lift.schemas import DatasetContext, LLMConfig


# We test only the deterministic helpers; no LLM calls needed.
class _FakeProfiler(DataProfiler):
    def __init__(self):
        # Skip LLM initialisation
        self.config = None
        self._client = None
        self._backend = None


@pytest.fixture
def profiler():
    return _FakeProfiler()


@pytest.fixture
def xi():
    return DatasetContext(
        domain="test",
        outcome_col="y",
        protected_col="race",
        cohort_notes="test cohort",
    )


# ---------------------------------------------------------------------------
# test_imbalance_ratio
# ---------------------------------------------------------------------------

def test_imbalance_ratio_basic(profiler):
    """I_out = max/min class counts."""
    s = pd.Series([0, 0, 0, 1])   # counts: {0:3, 1:1} → ratio = 3.0
    result = profiler._imbalance(s)
    assert round(result, 2) == 3.00


def test_imbalance_ratio_balanced(profiler):
    s = pd.Series([0, 1, 0, 1])
    assert profiler._imbalance(s) == 1.0


def test_imbalance_ratio_single_class(profiler):
    s = pd.Series([1, 1, 1])
    assert profiler._imbalance(s) == 1.0


# ---------------------------------------------------------------------------
# test_completeness
# ---------------------------------------------------------------------------

def test_completeness_five_percent(profiler, xi):
    """C with known 5% missing fraction."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        rng.randn(200, 10),
        columns=[f"f{i}" for i in range(9)] + ["y"],
    )
    df["race"] = rng.choice(["A", "B"], size=200)
    # Introduce ~5% missing in feature columns
    n_missing = int(0.05 * 200 * 9)
    rows = rng.randint(0, 200, n_missing)
    cols = rng.randint(0, 9, n_missing)
    for r, c in zip(rows, cols):
        df.iloc[r, c] = np.nan

    stats = profiler._compute_stats(df, xi)
    assert abs(stats["C"] - 0.05) < 0.02  # within 2pp


def test_completeness_no_missing(profiler, xi):
    df = pd.DataFrame({"f1": [1, 2, 3], "y": [0, 1, 0], "race": ["A", "B", "A"]})
    stats = profiler._compute_stats(df, xi)
    assert stats["C"] == 0.0


# ---------------------------------------------------------------------------
# test_stats structure
# ---------------------------------------------------------------------------

def test_stats_keys(profiler, xi):
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [5.0, 6.0, 7.0, 8.0],
        "cat": ["a", "b", "a", "b"],
        "y": [0, 1, 0, 1],
        "race": ["X", "Y", "X", "Y"],
    })
    stats = profiler._compute_stats(df, xi)
    for key in ("N", "P", "C", "I_out", "I_pop", "P_num", "P_cat"):
        assert key in stats


def test_feature_type_counts(profiler, xi):
    df = pd.DataFrame({
        "n1": [1.0, 2.0],
        "n2": [3.0, 4.0],
        "cat": ["a", "b"],
        "y": [0, 1],
        "race": ["A", "B"],
    })
    stats = profiler._compute_stats(df, xi)
    assert stats["P_num"] == 2
    assert stats["P_cat"] == 1
