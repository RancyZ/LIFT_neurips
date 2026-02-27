"""
Prompt templates for the GovernanceReporter agent.
"""
from __future__ import annotations

from lift.schemas import DataProfile, DatasetContext

SYSTEM = """You are the LLM Governance Reporter in LIFT.
Synthesize lifecycle evaluation results into a governance report.
Return ONLY a single JSON object."""


def build_user_prompt(
    scores_table: str,
    profile: DataProfile,
    xi: DatasetContext,
) -> str:
    return f"""Clinical ML governance report.
Domain: {xi.domain} | Protected: {xi.protected_col} | N={profile.N}, P={profile.P}

Scores (0-1, higher=better):
{scores_table}

Return ONLY this JSON:
{{
  "primary_model": "<model_id>",
  "primary_rationale": "<1-2 sentences>",
  "alternatives": [{{"model": "<id>", "rationale": "<brief>"}}],
  "avoid": [{{"model": "<id>", "reason": "<brief>"}}],
  "improvement_actions": {{"<model_id>": ["<action>"]}},
  "stage_interpretations": {{"<stage_id>": "<1 sentence>"}}
}}"""
