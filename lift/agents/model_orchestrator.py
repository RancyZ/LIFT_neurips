"""
ModelOrchestrator — Agent 2 (Φ_model).
Uses an LLM rubric to score candidate models and select lifecycle stages.

Paper reference: Equation 2.
  Φ_model(g(P_D, ξ_model)) = ({m_i}_{i∈I_D}, {l_j}_{j∈J_D})

Composite score: ẽ = Σ w_k * r_{i,k}  (Eq. 4)
  Weights w_k are adaptive based on dataset-specific risks in P_D.
  DatasetFlags are computed deterministically from P_D so the LLM
  receives explicit, auditable activation guidance.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from lift.agents.base_agent import BaseAgent
from lift.prompts import orchestrator_prompt as op_prompt
from lift.schemas import (
    DataProfile,
    DatasetFlags,
    LLMConfig,
    ModelLifecycleRoute,
    ModelScore,
    OrchestratorDecision,
    STAGE_SUB_DIMENSIONS,
    StageEmphasis,
)

logger = logging.getLogger(__name__)


class ModelOrchestrator(BaseAgent):
    VALID_STAGES = [
        "l1_learning_opt",
        "l2_generalizability",
        "l3_deployment",
        "l4_monitoring",
    ]
    ALL_MODELS = ["LR", "BR", "KNN", "SVM", "DT", "RF", "MLP", "RN", "VAE"]

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, profile: DataProfile) -> OrchestratorDecision:
        """
        Computes DatasetFlags deterministically, then calls LLM with the
        full rubric prompt (including flags and activation rules).
        Returns OrchestratorDecision with per-model routes.
        """
        flags = DatasetFlags.from_profile(profile)
        user_prompt = op_prompt.build_user_prompt(profile, flags, self.ALL_MODELS)
        raw = self.call_llm(
            prompt=user_prompt,
            system_prompt=op_prompt.SYSTEM,
            expect_json=True,
        )
        return self._validate_decision(raw, profile, flags)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_decision(
        self,
        raw: dict,
        profile: DataProfile,
        flags: DatasetFlags | None = None,
    ) -> OrchestratorDecision:
        """
        Parses LLM JSON output.
        - Filters unknown model/stage names.
        - Computes composite score from reported weights (Eq. 4); falls back
          to LLM-reported composite if weights are absent.
        - DatasetFlags computed deterministically if not supplied.
        """
        if flags is None:
            flags = DatasetFlags.from_profile(profile)

        orchestrator_summary = str(raw.get("orchestrator_summary", ""))
        raw_routes: List[dict] = raw.get("model_routes", [])

        model_routes: List[ModelLifecycleRoute] = []
        for r in raw_routes:
            mid = r.get("model_id", "")
            if mid not in self.ALL_MODELS:
                logger.warning("Orchestrator returned unknown model '%s' (skipped).", mid)
                continue

            raw_score: dict = r.get("model_score", {})

            # Clamp rubric scores to [0, 1]
            r1 = float(max(0.0, min(1.0, raw_score.get("r1", 0.5))))
            r2 = float(max(0.0, min(1.0, raw_score.get("r2", 0.5))))
            r3 = float(max(0.0, min(1.0, raw_score.get("r3", 0.5))))
            r4 = float(max(0.0, min(1.0, raw_score.get("r4", 0.5))))

            # Compute composite from adaptive weights (Eq. 4) if provided,
            # otherwise fall back to LLM-reported composite.
            w1 = raw_score.get("w1")
            w2 = raw_score.get("w2")
            w3 = raw_score.get("w3")
            w4 = raw_score.get("w4")
            if all(w is not None for w in (w1, w2, w3, w4)):
                w1, w2, w3, w4 = float(w1), float(w2), float(w3), float(w4)
                total = w1 + w2 + w3 + w4
                if total > 0:
                    w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
                composite = float(max(0.0, min(1.0, w1*r1 + w2*r2 + w3*r3 + w4*r4)))
            else:
                composite = float(max(0.0, min(1.0, raw_score.get("composite", 0.5))))

            ms = ModelScore(
                model_id=mid,
                r1_compatibility=r1,
                r2_complexity=r2,
                r3_interpretability=r3,
                r4_compute=r4,
                composite_score=composite,
                justification=str(raw_score.get("justification", "")),
            )

            raw_stages: dict = r.get("activated_stages", {})
            activated_stages: Dict[str, StageEmphasis] = {}
            for stage_id, emphasis_raw in raw_stages.items():
                if stage_id not in self.VALID_STAGES:
                    logger.warning("Model '%s': invalid stage '%s' (skipped).", mid, stage_id)
                    continue
                valid_sub_dims = STAGE_SUB_DIMENSIONS[stage_id]
                sub_emphasis = {
                    k: float(v)
                    for k, v in emphasis_raw.get("sub_dimension_emphasis", {}).items()
                    if k in valid_sub_dims
                }
                activated_stages[stage_id] = StageEmphasis(
                    sub_dimension_emphasis=sub_emphasis,
                    justification=str(emphasis_raw.get("justification", "")),
                )

            model_routes.append(ModelLifecycleRoute(
                model_id=mid,
                model_score=ms,
                activated_stages=activated_stages,
            ))

        return OrchestratorDecision(
            model_routes=model_routes,
            dataset_flags=flags,
            orchestrator_summary=orchestrator_summary,
        )
