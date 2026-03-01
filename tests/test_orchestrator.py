"""Unit tests for ModelOrchestrator validation logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.agents.model_orchestrator import ModelOrchestrator
from lift.schemas import DataProfile, LLMConfig


class _FakeOrchestrator(ModelOrchestrator):
    def __init__(self):
        self.config = None
        self._client = None
        self._backend = None


@pytest.fixture
def orc():
    return _FakeOrchestrator()


@pytest.fixture
def profile():
    """Minimal DataProfile for deterministic DatasetFlags computation."""
    return DataProfile(
        N=1000, P=20, C=0.05,
        P_num=15, P_cat=5,
        I_out=1.2, I_pop=1.1,
        domain="test", protected_attr="sex", outcome_col="y",
    )


# ---------------------------------------------------------------------------
# test_valid_stages
# ---------------------------------------------------------------------------

def test_valid_stages_all_valid(orc, profile):
    raw = {
        "model_routes": [
            {
                "model_id": "LR",
                "model_score": {"r1": 0.8, "r2": 0.9, "r3": 0.9, "r4": 0.9, "composite": 0.88, "justification": "ok"},
                "activated_stages": {
                    "l1_learning_opt": {"sub_dimension_emphasis": {"efficiency": 0.8}, "justification": "reason"},
                    "l2_generalizability": {"sub_dimension_emphasis": {"discriminative_power": 0.7}, "justification": "reason"},
                },
            }
        ],
        "orchestrator_summary": "summary",
    }
    decision = orc._validate_decision(raw, profile)
    all_stages = decision.globally_activated_stages()
    assert all(s in orc.VALID_STAGES for s in all_stages)


def test_valid_stages_filters_invalid(orc, profile):
    raw = {
        "model_routes": [
            {
                "model_id": "LR",
                "model_score": {"r1": 0.8, "r2": 0.9, "r3": 0.9, "r4": 0.9, "composite": 0.88, "justification": "ok"},
                "activated_stages": {
                    "l1_learning_opt": {"sub_dimension_emphasis": {"efficiency": 0.8}, "justification": "reason"},
                    "l5_nonexistent": {"sub_dimension_emphasis": {}, "justification": "bad"},
                },
            }
        ],
        "orchestrator_summary": "summary",
    }
    decision = orc._validate_decision(raw, profile)
    all_stages = decision.globally_activated_stages()
    assert "l5_nonexistent" not in all_stages
    assert "l1_learning_opt" in all_stages


def test_valid_stages_empty(orc, profile):
    raw = {"model_routes": [], "orchestrator_summary": ""}
    decision = orc._validate_decision(raw, profile)
    assert decision.globally_activated_stages() == []


# ---------------------------------------------------------------------------
# test_rubric_bounds
# ---------------------------------------------------------------------------

def test_rubric_bounds_valid(orc, profile):
    """All r1..r4 must be clamped to [0,1]."""
    raw = {
        "model_routes": [
            {
                "model_id": "LR",
                "model_score": {"r1": 0.8, "r2": 0.6, "r3": 0.7, "r4": 0.9, "composite": 0.75, "justification": "good"},
                "activated_stages": {},
            }
        ],
        "orchestrator_summary": "summary",
    }
    decision = orc._validate_decision(raw, profile)
    ms = decision.model_routes[0].model_score
    for score in (ms.r1_compatibility, ms.r2_complexity, ms.r3_interpretability, ms.r4_compute):
        assert 0 <= score <= 1


def test_rubric_bounds_clamped(orc, profile):
    """Scores outside [0,1] must be clamped."""
    raw = {
        "model_routes": [
            {
                "model_id": "RF",
                "model_score": {"r1": 1.5, "r2": -0.1, "r3": 0.5, "r4": 2.0, "composite": 0.5, "justification": "test"},
                "activated_stages": {},
            }
        ],
        "orchestrator_summary": "summary",
    }
    decision = orc._validate_decision(raw, profile)
    ms = decision.model_routes[0].model_score
    assert ms.r1_compatibility == 1.0
    assert ms.r2_complexity == 0.0
    assert ms.r4_compute == 1.0


def test_unknown_model_filtered(orc, profile):
    raw = {
        "model_routes": [
            {
                "model_id": "XGBOOST",  # not in LIFT
                "model_score": {"r1": 0.5, "r2": 0.5, "r3": 0.5, "r4": 0.5, "composite": 0.5, "justification": "nope"},
                "activated_stages": {},
            }
        ],
        "orchestrator_summary": "summary",
    }
    decision = orc._validate_decision(raw, profile)
    assert len(decision.model_routes) == 0
