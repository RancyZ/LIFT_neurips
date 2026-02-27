"""
ModelOrchestrator — Agent 2 (Φ_model).
Uses an LLM rubric to score candidate models and select lifecycle stages.
"""
from __future__ import annotations

import logging
from typing import List

from lift.agents.base_agent import BaseAgent
from lift.prompts import orchestrator_prompt as op_prompt
from lift.schemas import (
    DataProfile,
    LLMConfig,
    ModelScore,
    OrchestratorDecision,
)

logger = logging.getLogger(__name__)


class ModelOrchestrator(BaseAgent):
    VALID_STAGES = [
        "l1_learning_opt",
        "l2_generalizability",
        "l3_deployment",
        "l4_monitoring",
    ]
    ALL_MODELS = ["LR", "BR", "KNN", "SVM", "DT", "RF", "MLP", "ResNet", "VAE"]

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, profile: DataProfile) -> OrchestratorDecision:
        """
        Calls LLM with structured rubric prompt.
        LLM scores each model on 4 criteria and selects lifecycle stages.
        Validates output: stages must be subset of VALID_STAGES.
        Returns OrchestratorDecision.
        """
        user_prompt = op_prompt.build_user_prompt(profile, self.ALL_MODELS)
        raw = self.call_llm(
            prompt=user_prompt,
            system_prompt=op_prompt.SYSTEM,
            expect_json=True,
        )
        return self._validate_decision(raw)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_decision(self, raw: dict) -> OrchestratorDecision:
        """Parses LLM JSON; filters invalid stage/model names."""
        raw_models: List[dict] = raw.get("selected_models", [])
        activated: List[str] = raw.get("activated_stages", [])
        stage_just: dict = raw.get("stage_justification", {})

        # Filter invalid stage names
        valid_activated = [s for s in activated if s in self.VALID_STAGES]
        if len(valid_activated) != len(activated):
            dropped = set(activated) - set(valid_activated)
            logger.warning("Orchestrator returned invalid stages (dropped): %s", dropped)

        # Filter invalid model names; clamp scores to [0,1]
        model_scores: List[ModelScore] = []
        for m in raw_models:
            mid = m.get("model_id", "")
            if mid not in self.ALL_MODELS:
                logger.warning("Orchestrator returned unknown model '%s' (skipped).", mid)
                continue
            ms = ModelScore(
                model_id=mid,
                r1_compatibility=float(max(0.0, min(1.0, m.get("r1", 0.5)))),
                r2_complexity=float(max(0.0, min(1.0, m.get("r2", 0.5)))),
                r3_interpretability=float(max(0.0, min(1.0, m.get("r3", 0.5)))),
                r4_compute=float(max(0.0, min(1.0, m.get("r4", 0.5)))),
                composite_score=float(max(0.0, min(1.0, m.get("composite", 0.5)))),
                justification=str(m.get("justification", "")),
            )
            model_scores.append(ms)

        return OrchestratorDecision(
            selected_models=model_scores,
            activated_stages=valid_activated,
            stage_justification={k: v for k, v in stage_just.items() if k in self.VALID_STAGES},
        )
