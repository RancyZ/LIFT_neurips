"""
Prompt templates for the GovernanceReporter agent.

Paper reference: Section 3.1.3, Equation 3.
  R_D ~ Φ_rep(g(E_D, ξ_rep))
  R_D = ({e_l_j}_{j∈J_D}, γ_D)

The reporter:
- Does NOT recompute metrics — only interprets them
- Synthesises cross-stage evidence holistically
- Produces stage interpretations {e_l_j} and recommendations γ_D
"""
from __future__ import annotations

from lift.schemas import DataProfile, DatasetContext, OrchestratorDecision

SYSTEM = """You are the LLM Governance Reporter in LIFT (Lifecycle-based Integrity & Fidelity Testing).
You synthesize lifecycle evaluation results into a structured, lifecycle-aware governance report.
Your goal is not merely to summarise performance metrics but to interpret evaluation outcomes
in the context of lifecycle integrity and fidelity, and recommend actionable strategies.
You do NOT alter evaluation results. You preserve transparency.
Return ONLY a single JSON object. No markdown, no explanation, no preamble."""

# Stage descriptions for LLM context
_STAGE_DESCRIPTIONS = {
    "l1_learning_opt":     "Learning & Optimisation — computational efficiency and training feasibility (lower training time = higher score)",
    "l2_generalizability": "Generalizability — discriminative power (Acc, Spec) and robustness under missing data",
    "l3_deployment":       "Deployment — operational robustness under resampling, subgroup parity (EOP/FPR/DP disparity), and explanation consistency across protected groups",
    "l4_monitoring":       "Monitoring & Updating — distribution shift detection via Wasserstein distance (lower drift = higher score)",
}


def build_user_prompt(
    scores_table: str,
    raw_metrics_block: str,
    profile: DataProfile,
    decision: OrchestratorDecision,
    xi: DatasetContext,
) -> str:
    # Activated stage descriptions
    activated = decision.globally_activated_stages()
    stage_desc = "\n".join(
        f"  {s}: {_STAGE_DESCRIPTIONS.get(s, s)}"
        for s in activated
    )

    # Dataset flags summary for contextual recommendations
    flags = decision.dataset_flags
    flag_lines = []
    if flags.high_dimensionality:
        flag_lines.append("HIGH_DIMENSIONALITY — overfitting risk; favour simpler models")
    if flags.high_incompleteness:
        flag_lines.append(f"HIGH_INCOMPLETENESS (C={profile.C:.3f}) — missing-data risk affects robustness")
    if flags.high_outcome_imbalance:
        flag_lines.append(f"HIGH_OUTCOME_IMBALANCE (I_out={profile.I_out:.2f}) — minority-class sensitivity is critical")
    if flags.high_population_imbalance:
        flag_lines.append(f"HIGH_POPULATION_IMBALANCE (I_pop={profile.I_pop:.2f}) — subgroup parity is critical")
    if flags.balanced_dataset:
        flag_lines.append("BALANCED DATASET — distribution shift over time is the primary long-term risk")
    flags_block = "\n".join(f"  • {l}" for l in flag_lines) if flag_lines else "  • No extreme flags"

    narrative_block = (
        f"\nData profile narrative: {profile.profile_narrative}"
        if profile.profile_narrative
        else ""
    )

    orchestrator_block = (
        f"\nOrchestrator reasoning: {decision.orchestrator_summary}"
        if decision.orchestrator_summary
        else ""
    )

    return f"""Governance report for: {xi.domain}
Outcome: {xi.outcome_col} | Protected attribute: {xi.protected_col}
Dataset: N={profile.N}, P={profile.P}, C={profile.C:.3f}, I_out={profile.I_out:.2f}, I_pop={profile.I_pop:.2f}{narrative_block}{orchestrator_block}

Dataset risk flags:
{flags_block}

Activated lifecycle stages:
{stage_desc}

Normalised lifecycle scores (0–1, higher = better across all stages):
{scores_table}

Key raw metrics per model-stage:
{raw_metrics_block}

Instructions:
- Select the PRIMARY model based on the strongest cross-stage lifecycle profile, not a single metric.
- List STRONG ALTERNATIVES with comparative lifecycle trade-offs.
- List models to AVOID, citing specific lifecycle weaknesses.
- For IMPROVEMENT ACTIONS, give concrete, stage-specific, actionable suggestions per model, organised by weakness type (e.g. imbalance handling, threshold tuning, fairness constraints, explanation consistency).
- For STAGE INTERPRETATIONS, write one paragraph per activated stage explaining what the scores reveal about the cohort's trustworthiness challenges.
- Ground every recommendation in the dataset risk flags and specific score values above.

Return ONLY this JSON:
{{
  "primary_model": "<model_id>",
  "primary_rationale": "<Detailed justification citing cross-stage evidence and dataset context>",
  "alternatives": [
    {{"model": "<id>", "rationale": "<Comparative lifecycle trade-off analysis>"}}
  ],
  "avoid": [
    {{"model": "<id>", "reason": "<Specific lifecycle weaknesses with score references>"}}
  ],
  "improvement_actions": {{
    "<model_id>": [
      "<Concrete action 1 tied to a specific weak stage or metric>",
      "<Concrete action 2>"
    ]
  }},
  "stage_interpretations": {{
    "<stage_id>": "<Paragraph explaining what scores reveal about lifecycle integrity for this stage>"
  }}
}}"""
