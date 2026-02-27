"""
Shared data schemas (dataclasses) used as contracts between pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 4.1  DataProfile  (P_D) — output of DataProfiler
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    # Population Integrity
    N: int                  # total samples
    P: int                  # total predictive features (excl. y, a)
    C: float                # incompleteness rate ∈ [0,1]

    # Feature Characteristics
    P_num: int              # number of numerical features
    P_cat: int              # number of categorical features

    # Data Bias
    I_out: float            # outcome imbalance ratio ≥ 1
    I_pop: float            # population group imbalance ratio ≥ 1

    # Context (passed through)
    domain: str
    protected_attr: str
    outcome_col: str


# ---------------------------------------------------------------------------
# 4.2  OrchestratorDecision — output of ModelOrchestrator
# ---------------------------------------------------------------------------

@dataclass
class ModelScore:
    model_id: str
    r1_compatibility: float     # data-model fit ∈ [0,1]
    r2_complexity: float        # overfitting risk ∈ [0,1]
    r3_interpretability: float
    r4_compute: float
    composite_score: float
    justification: str


@dataclass
class OrchestratorDecision:
    selected_models: List[ModelScore]
    activated_stages: List[str]              # subset of [l1,l2,l3,l4]
    stage_justification: Dict[str, str]      # stage_id → reason


# ---------------------------------------------------------------------------
# 4.3  EvalResults  (E_D)
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage_id: str
    model_id: str
    score: float                # normalised ∈ [0,1]
    raw: Dict[str, Any] = field(default_factory=dict)


# E_D is a dict keyed (model_id, stage_id) → StageResult
EvalResults = Dict[Tuple[str, str], StageResult]


# ---------------------------------------------------------------------------
# 4.4  GovReport  (R_D)
# ---------------------------------------------------------------------------

@dataclass
class GovReport:
    primary_model: str
    primary_rationale: str
    alternatives: List[Dict[str, str]]              # [{model, rationale}]
    avoid: List[Dict[str, str]]                     # [{model, reason}]
    improvement_actions: Dict[str, List[str]]       # model → [actions]
    stage_interpretations: Dict[str, str]           # stage → narrative
    raw_llm_output: str = ""


# ---------------------------------------------------------------------------
# 5.1  DatasetContext — caller-supplied dataset metadata
# ---------------------------------------------------------------------------

@dataclass
class DatasetContext:
    domain: str
    outcome_col: str
    protected_col: str
    cohort_notes: str
    operational_constraints: Optional[str] = None


# ---------------------------------------------------------------------------
# LLMConfig — extracted from config.yaml llm section
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model: str
    temperature: float
    max_tokens: int
    max_retries: int
