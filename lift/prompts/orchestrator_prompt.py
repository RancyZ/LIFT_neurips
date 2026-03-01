"""
Prompt templates for the ModelOrchestrator agent.
"""
from __future__ import annotations

from typing import List

from lift.schemas import DataProfile, DatasetFlags

SYSTEM = """You are the LLM Model Orchestrator in LIFT (Lifecycle-based Integrity & Fidelity Testing).
Return ONLY a single JSON object. No markdown or explanation."""


def build_user_prompt(
    profile: DataProfile,
    flags: DatasetFlags,
    all_models: List[str],
) -> str:
    models_str = ", ".join(all_models)

    # ── Translate deterministic flags into plain-language guidance ──────────
    flag_lines: List[str] = []
    if flags.high_dimensionality:
        flag_lines.append(
            "HIGH_DIMENSIONALITY (N/P < 10): high overfitting risk — "
            "prefer simpler models (LR, BR, SVM, DT, RF); penalise MLP/ResNet/VAE. "
            "Activate l1_learning_opt and l2_generalizability."
        )
    if flags.high_incompleteness:
        flag_lines.append(
            "HIGH_INCOMPLETENESS (C > 0.20): missing-data risk — "
            "activate l2_generalizability (missing_data_robustness) and "
            "l3_deployment (robustness)."
        )
    if flags.high_outcome_imbalance:
        flag_lines.append(
            "HIGH_OUTCOME_IMBALANCE (I_out > 3): rare-event prediction risk — "
            "exclude KNN (biased toward majority class) and DT (majority-class bias). "
            "Activate l3_deployment with high subgroup_parity and explainability emphasis."
        )
    if flags.high_population_imbalance:
        flag_lines.append(
            "HIGH_POPULATION_IMBALANCE (I_pop > 3): subgroup undercoverage risk — "
            "activate l3_deployment with high subgroup_parity emphasis (EOP, FPR, DP checks)."
        )
    if flags.balanced_dataset:
        flag_lines.append(
            "BALANCED (I_out ≤ 1.5, I_pop ≤ 1.5): low immediate bias risk — "
            "activate l4_monitoring to detect real-world distribution shift."
        )
    if not flag_lines:
        flag_lines.append("No extreme dataset flags — apply standard evaluation coverage.")

    flags_block = "\n".join(f"  • {l}" for l in flag_lines)

    # ── Rubric weight guidance based on flags ───────────────────────────────
    # Weights must sum to 1.0; adjust emphasis per paper Section 3.2
    weight_guidance = "Equal weights (0.25 each) are a baseline. Adjust:"
    if flags.high_dimensionality:
        weight_guidance += " increase w2 (complexity) when N/P is small;"
    if flags.high_outcome_imbalance or flags.high_population_imbalance:
        weight_guidance += " increase w3 (interpretability) when imbalance is high;"
    if flags.high_incompleteness:
        weight_guidance += " increase w1 (data-model fit) when C is high;"

    return f"""Clinical dataset: {profile.domain}
N={profile.N}, P={profile.P}, C={profile.C:.3f}, I_out={profile.I_out:.2f}, I_pop={profile.I_pop:.2f}
Numerical features: {profile.P_num}, Categorical features: {profile.P_cat}
Outcome column: {profile.outcome_col}, Protected attribute: {profile.protected_attr}
Candidate models: {models_str}

DATASET FLAGS (deterministically computed — you must respect these):
{flags_block}

RUBRIC — score each SELECTED model on four criteria ∈ [0,1]:
  r1 = data-model compatibility (feature types, dimensionality, sample size)
  r2 = low complexity / overfitting risk (higher = simpler / better regularised)
  r3 = clinical interpretability and explanation readiness
  r4 = low computational and maintenance burden

Composite score: weighted sum w1*r1 + w2*r2 + w3*r3 + w4*r4 (weights sum to 1.0).
{weight_guidance}

STAGE ACTIVATION — for each selected model, activate only the stages warranted by the flags above.
Sub-dimension emphasis weights ∈ [0,1] (higher = more scrutiny):
  l1_learning_opt:     efficiency
  l2_generalizability: discriminative_power, missing_data_robustness
  l3_deployment:       robustness, subgroup_parity, explainability
  l4_monitoring:       drift_detection

Return ONLY this JSON (no extra text):
{{
  "model_routes": [
    {{
      "model_id": "LR",
      "model_score": {{
        "r1": 0.85, "r2": 0.90, "r3": 0.95, "r4": 0.90,
        "w1": 0.20, "w2": 0.30, "w3": 0.30, "w4": 0.20,
        "composite": 0.90,
        "justification": "brief reason for selection and weights"
      }},
      "activated_stages": {{
        "l2_generalizability": {{
          "sub_dimension_emphasis": {{"discriminative_power": 0.8, "missing_data_robustness": 0.6}},
          "justification": "reason tied to dataset flags"
        }},
        "l3_deployment": {{
          "sub_dimension_emphasis": {{"robustness": 0.7, "subgroup_parity": 0.9, "explainability": 0.8}},
          "justification": "reason tied to dataset flags"
        }}
      }}
    }}
  ],
  "orchestrator_summary": "Concise narrative: which models were selected/excluded and why, which stages activated and why, referencing dataset flags."
}}"""
