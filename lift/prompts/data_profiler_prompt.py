"""
Prompt templates for the DataProfiler agent.
"""
from __future__ import annotations

from lift.schemas import DatasetContext

SYSTEM = """You are the LLM Data Profiler in the LIFT framework.
Return ONLY a single JSON object matching the schema below.
No markdown, no explanation, no preamble."""


def build_user_prompt(stats: dict, xi: DatasetContext) -> str:
    return f"""Dataset: {xi.domain} | outcome={xi.outcome_col} | protected={xi.protected_col}
Stats: N={stats['N']}, P={stats['P']}, C={stats['C']:.4f}, I_out={stats['I_out']:.4f}, I_pop={stats['I_pop']:.4f}, P_num={stats['P_num']}, P_cat={stats['P_cat']}

Return ONLY this JSON:
{{"population_integrity":{{"N":{stats['N']},"P":{stats['P']},"C":{stats['C']:.4f}}},"feature_characteristics":{{"P_num":{stats['P_num']},"P_cat":{stats['P_cat']}}},"data_bias":{{"I_out":{{"value":{stats['I_out']:.4f}}},"I_pop":{{"value":{stats['I_pop']:.4f}}}}}}}"""
