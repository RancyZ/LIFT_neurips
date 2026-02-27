"""
Prompt templates for the ModelOrchestrator agent.
"""
from __future__ import annotations

from typing import List

from lift.schemas import DataProfile

SYSTEM = """You are the LLM Model Orchestrator in LIFT.
Return ONLY a single JSON object. No markdown or explanation."""


def build_user_prompt(profile: DataProfile, all_models: List[str]) -> str:
    models_str = ", ".join(all_models)
    return f"""Clinical dataset: {profile.domain}
N={profile.N}, P={profile.P}, C={profile.C:.3f}, I_out={profile.I_out:.2f}, I_pop={profile.I_pop:.2f}
Outcome: {profile.outcome_col}, Protected: {profile.protected_attr}
Candidates: {models_str}

Score each model (0-1): r1=data fit, r2=low overfit risk, r3=interpretability, r4=low compute. composite=mean.
Select stages from: l1_learning_opt, l2_generalizability, l3_deployment, l4_monitoring.

Return ONLY this JSON (no extra text):
{{
  "selected_models": [
    {{"model_id": "LR", "r1": 0.8, "r2": 0.9, "r3": 0.9, "r4": 0.9, "composite": 0.88, "justification": "brief"}}
  ],
  "activated_stages": ["l1_learning_opt", "l2_generalizability"],
  "stage_justification": {{"l1_learning_opt": "reason", "l2_generalizability": "reason"}}
}}"""
