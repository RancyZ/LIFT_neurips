"""
Prompt templates for the DataProfiler agent.

All numeric metrics (C, I_out, I_pop, N, P, P_num, P_cat) are computed
deterministically in DataProfiler._compute_stats() before this prompt is called.
The LLM's role is to interpret and contextualise those metrics in clinical terms
— not to recompute them.  (Paper Section 3.1.1)
"""
from __future__ import annotations

from lift.schemas import DatasetContext

SYSTEM = """You are the LLM Data Profiler in the LIFT framework (Lifecycle-based Integrity & Fidelity Testing).
You will receive pre-computed dataset statistics and clinical context.
Your role is to interpret these statistics in clinical terms, identifying key risks
and dataset characteristics that will guide model selection and lifecycle evaluation.
Return ONLY a single JSON object. No markdown, no explanation, no preamble."""


def build_user_prompt(stats: dict, xi: DatasetContext) -> str:
    S = stats["N"] * stats["P"]  # Dataset scale S = N × P (Section 3.2.1)

    constraints = (
        f"\nOperational constraints: {xi.operational_constraints}"
        if xi.operational_constraints
        else ""
    )

    return f"""Clinical domain: {xi.domain}
Cohort notes: {xi.cohort_notes}{constraints}
Outcome column: {xi.outcome_col} | Protected attribute: {xi.protected_col}

Pre-computed dataset statistics (P_D = C, I_out, I_pop, T, S):

  Population Integrity:
    N = {stats['N']}   (number of instances)
    P = {stats['P']}   (predictive features, excluding outcome and protected attribute)
    S = {S}   (dataset scale = N × P; small N relative to P → high overfitting risk)
    C = {stats['C']:.4f}   (incompleteness: fraction of missing feature values ∈ [0,1])

  Feature Characteristics (T):
    P_num = {stats['P_num']}   (numerical features)
    P_cat = {stats['P_cat']}   (categorical features)

  Data Bias:
    I_out = {stats['I_out']:.4f}   (outcome imbalance = max_class_count / min_class_count; 1.0 = perfectly balanced)
    I_pop = {stats['I_pop']:.4f}   (population imbalance = max_group_count / min_group_count; 1.0 = perfectly balanced)

Interpret these statistics in clinical context. Identify:
- Key integrity risks (missingness, scale, dimensionality)
- Outcome imbalance implications for minority-class sensitivity
- Population imbalance implications for subgroup fairness
- Feature composition implications for model selection

Return ONLY this JSON:
{{
  "profile_narrative": "2-4 sentence clinical interpretation of the dataset profile, referencing specific values of C, I_out, I_pop, and N/P to explain the key risks and evaluation priorities for this dataset."
}}"""
