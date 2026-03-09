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

    # LLM-generated clinical interpretation of the profile
    profile_narrative: str = ""


# ---------------------------------------------------------------------------
# 4.2  OrchestratorDecision — output of ModelOrchestrator
# ---------------------------------------------------------------------------

@dataclass
class ModelScore:
    model_id: str
    r1_compatibility: float     # data-model fit ∈ [0,1]
    r2_complexity: float        # overfitting risk ∈ [0,1]
    r3_interpretability: float  # interpretability ∈ [0,1]
    r4_compute: float           # low compute burden ∈ [0,1]
    composite_score: float      # weighted aggregate ∈ [0,1]
    justification: str

    def __post_init__(self) -> None:
        for attr in ("r1_compatibility", "r2_complexity",
                     "r3_interpretability", "r4_compute", "composite_score"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"ModelScore.{attr}={val} must be in [0, 1].")


@dataclass
class DatasetFlags:
    """
    Deterministic flags derived from DataProfile (Section 3.2.1).
    Drives adaptive stage activation and sub-dimension emphasis.
    Computed in code, not by the LLM, so reasoning is auditable.
    """
    high_incompleteness: bool       # C > threshold → favour robust models, activate l2
    high_dimensionality: bool       # N/P < threshold → drop neural nets, activate l1, l2, l3
    large_n: bool                   # N > threshold → include VAE/ResNet, activate l1
    high_outcome_imbalance: bool    # I_out >> 1 → drop KNN/DT, activate l3 parity
    high_population_imbalance: bool # I_pop >> 1 → activate l3 parity
    balanced_dataset: bool          # I_out & I_pop ≈ 1 → activate l4
    mixed_features: bool            # P_num > 0 and P_cat > 0 → include tree + linear models

    # Thresholds stored for traceability
    incompleteness_threshold: float = 0.2
    dim_ratio_threshold: float = 10.0
    large_n_threshold: int = 5000
    imbalance_threshold: float = 3.0
    balanced_threshold: float = 1.5

    @classmethod
    def from_profile(
        cls,
        profile: DataProfile,
        incompleteness_threshold: float = 0.2,
        dim_ratio_threshold: float = 10.0,
        large_n_threshold: int = 5000,
        imbalance_threshold: float = 3.0,
        balanced_threshold: float = 1.5,
    ) -> "DatasetFlags":
        """Deterministically derives flags from a DataProfile."""
        return cls(
            high_incompleteness=profile.C > incompleteness_threshold,
            high_dimensionality=(profile.N / profile.P) < dim_ratio_threshold,
            large_n=profile.N > large_n_threshold,
            high_outcome_imbalance=profile.I_out > imbalance_threshold,
            high_population_imbalance=profile.I_pop > imbalance_threshold,
            balanced_dataset=(
                profile.I_out <= balanced_threshold and
                profile.I_pop <= balanced_threshold
            ),
            mixed_features=(profile.P_num > 0 and profile.P_cat > 0),
            incompleteness_threshold=incompleteness_threshold,
            dim_ratio_threshold=dim_ratio_threshold,
            large_n_threshold=large_n_threshold,
            imbalance_threshold=imbalance_threshold,
            balanced_threshold=balanced_threshold,
        )


# Valid sub-dimensions per stage — used for validation in orchestrator
STAGE_SUB_DIMENSIONS: Dict[str, Set[str]] = {
    "l1_learning_opt":    {"efficiency"},
    "l2_generalizability": {"discriminative_power", "missing_data_robustness"},
    "l3_deployment":      {"robustness", "subgroup_parity", "explainability"},
    "l4_monitoring":      {"drift_detection"},
}

VALID_STAGES: Set[str] = set(STAGE_SUB_DIMENSIONS.keys())


@dataclass
class StageEmphasis:
    """
    Per-stage emphasis config for a single model.
    Captures that stages are not binary on/off but carry
    varying emphasis weights per sub-dimension (Section 3.2.1).

    sub_dimension_emphasis keys must match STAGE_SUB_DIMENSIONS
    for the corresponding stage.
    """
    sub_dimension_emphasis: Dict[str, float]    # sub-dim → weight ∈ [0,1]
    justification: str

    def __post_init__(self) -> None:
        self.sub_dimension_emphasis = {
            k: max(0.0, min(1.0, v))
            for k, v in self.sub_dimension_emphasis.items()
        }


@dataclass
class ModelLifecycleRoute:
    """
    Per-model lifecycle routing (Section 3.2.2).
    Each dataset-model pair (D, m) is routed through an
    adaptively selected subset of stages with emphasis weights.
    """
    model_id: str
    model_score: ModelScore
    activated_stages: Dict[str, StageEmphasis] = field(default_factory=dict)

    def __post_init__(self) -> None:
        invalid = set(self.activated_stages.keys()) - VALID_STAGES
        if invalid:
            raise ValueError(f"Invalid stage keys for model '{self.model_id}': {invalid}")


@dataclass
class OrchestratorDecision:
    """
    Full orchestrator output aligned with LIFT paper.

    Changes from original:
    - selected_models → model_routes: stage routing is now per-model
    - activated_stages global list removed; lives inside each ModelLifecycleRoute
    - stage_justification removed; lives inside each StageEmphasis
    - dataset_flags added for deterministic, auditable activation reasoning
    - orchestrator_summary added for high-level LLM narrative
    """
    model_routes: List[ModelLifecycleRoute]
    dataset_flags: DatasetFlags
    orchestrator_summary: str

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def globally_activated_stages(self) -> List[str]:
        """
        Union of all activated stages across models.
        Used by pipeline runners to know which executors to instantiate.
        """
        stages: Set[str] = set()
        for route in self.model_routes:
            stages.update(route.activated_stages.keys())
        return sorted(stages)

    def routes_for_stage(self, stage: str) -> List[ModelLifecycleRoute]:
        """All model routes that include a given stage."""
        return [r for r in self.model_routes if stage in r.activated_stages]

    def emphasis_for(self, model_id: str, stage: str) -> Optional[StageEmphasis]:
        """Emphasis config for a specific model-stage pair."""
        for route in self.model_routes:
            if route.model_id == model_id:
                return route.activated_stages.get(stage)
        return None

    def selected_model_ids(self) -> List[str]:
        """Convenience: list of all selected model IDs."""
        return [r.model_id for r in self.model_routes]


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
    id_col: Optional[str] = None
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