"""Unit tests for ModelOrchestrator validation logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.agents.model_orchestrator import ModelOrchestrator
from lift.schemas import LLMConfig


class _FakeOrchestrator(ModelOrchestrator):
    def __init__(self):
        self.config = None
        self._client = None
        self._backend = None


@pytest.fixture
def orc():
    return _FakeOrchestrator()


# ---------------------------------------------------------------------------
# test_valid_stages
# ---------------------------------------------------------------------------

def test_valid_stages_all_valid(orc):
    raw = {
        "selected_models": [],
        "activated_stages": ["l1_learning_opt", "l2_generalizability"],
        "stage_justification": {},
    }
    decision = orc._validate_decision(raw)
    assert all(s in orc.VALID_STAGES for s in decision.activated_stages)


def test_valid_stages_filters_invalid(orc):
    raw = {
        "selected_models": [],
        "activated_stages": ["l1_learning_opt", "l5_nonexistent"],
        "stage_justification": {},
    }
    decision = orc._validate_decision(raw)
    assert "l5_nonexistent" not in decision.activated_stages
    assert "l1_learning_opt" in decision.activated_stages


def test_valid_stages_empty(orc):
    raw = {"selected_models": [], "activated_stages": [], "stage_justification": {}}
    decision = orc._validate_decision(raw)
    assert decision.activated_stages == []


# ---------------------------------------------------------------------------
# test_rubric_bounds
# ---------------------------------------------------------------------------

def test_rubric_bounds_valid(orc):
    """All r1..r4 must be clamped to [0,1]."""
    raw = {
        "selected_models": [
            {
                "model_id": "LR",
                "r1": 0.8, "r2": 0.6, "r3": 0.7, "r4": 0.9,
                "composite": 0.75, "justification": "good",
            }
        ],
        "activated_stages": [],
        "stage_justification": {},
    }
    decision = orc._validate_decision(raw)
    ms = decision.selected_models[0]
    for score in (ms.r1_compatibility, ms.r2_complexity, ms.r3_interpretability, ms.r4_compute):
        assert 0 <= score <= 1


def test_rubric_bounds_clamped(orc):
    """Scores outside [0,1] must be clamped."""
    raw = {
        "selected_models": [
            {
                "model_id": "RF",
                "r1": 1.5, "r2": -0.1, "r3": 0.5, "r4": 2.0,
                "composite": 0.5, "justification": "test",
            }
        ],
        "activated_stages": [],
        "stage_justification": {},
    }
    decision = orc._validate_decision(raw)
    ms = decision.selected_models[0]
    assert ms.r1_compatibility == 1.0
    assert ms.r2_complexity == 0.0
    assert ms.r4_compute == 1.0


def test_unknown_model_filtered(orc):
    raw = {
        "selected_models": [
            {
                "model_id": "XGBOOST",  # not in LIFT
                "r1": 0.5, "r2": 0.5, "r3": 0.5, "r4": 0.5,
                "composite": 0.5, "justification": "nope",
            }
        ],
        "activated_stages": [],
        "stage_justification": {},
    }
    decision = orc._validate_decision(raw)
    assert len(decision.selected_models) == 0
