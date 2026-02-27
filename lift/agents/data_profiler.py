"""
DataProfiler — Agent 1 (Φ_data).
Computes deterministic stats from a DataFrame, then calls the LLM
to produce a structured DataProfile (P_D).
"""
from __future__ import annotations

import pandas as pd

from lift.agents.base_agent import BaseAgent
from lift.prompts import data_profiler_prompt as dp_prompt
from lift.schemas import DataProfile, DatasetContext, LLMConfig


class DataProfiler(BaseAgent):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        df: pd.DataFrame,
        xi: DatasetContext,
    ) -> DataProfile:
        """
        Step 1: Compute deterministic stats from df.
        Step 2: Build JSON-schema prompt from stats + xi.
        Step 3: Call LLM → parse structured JSON → return DataProfile.
        """
        stats = self._compute_stats(df, xi)
        user_prompt = dp_prompt.build_user_prompt(stats, xi)
        result = self.call_llm(
            prompt=user_prompt,
            system_prompt=dp_prompt.SYSTEM,
            expect_json=True,
        )
        return self._parse_profile(result, xi)

    # ------------------------------------------------------------------
    # Deterministic statistics
    # ------------------------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame, xi: DatasetContext) -> dict:
        """
        Returns: { N, P, C, I_out, I_pop, P_num, P_cat }

        C = 1 - (observed_cells / total_cells)
        I_out = max(class_counts) / min(class_counts)
        I_pop = max(group_counts) / min(group_counts)
        """
        # Feature columns (exclude outcome and protected)
        feature_cols = [
            c for c in df.columns
            if c not in (xi.outcome_col, xi.protected_col)
        ]

        N = len(df)
        P = len(feature_cols)

        # Incompleteness
        total_cells = N * len(df.columns)
        observed_cells = df.notna().sum().sum()
        C = round(1.0 - observed_cells / total_cells, 6) if total_cells > 0 else 0.0

        # Imbalance
        I_out = self._imbalance(df[xi.outcome_col])
        I_pop = self._imbalance(df[xi.protected_col])

        # Feature types
        feature_df = df[feature_cols]
        P_num = int(feature_df.select_dtypes(include="number").shape[1])
        P_cat = int(feature_df.select_dtypes(exclude="number").shape[1])

        return {
            "N": N,
            "P": P,
            "C": C,
            "I_out": I_out,
            "I_pop": I_pop,
            "P_num": P_num,
            "P_cat": P_cat,
        }

    def _imbalance(self, series: pd.Series) -> float:
        counts = series.dropna().value_counts()
        if len(counts) < 2:
            return 1.0
        return round(counts.max() / counts.min(), 4)

    # ------------------------------------------------------------------
    # Parse LLM JSON → DataProfile
    # ------------------------------------------------------------------

    def _parse_profile(self, raw: dict, xi: DatasetContext) -> DataProfile:
        pi = raw.get("population_integrity", {})
        fc = raw.get("feature_characteristics", {})
        db = raw.get("data_bias", {})

        return DataProfile(
            N=int(pi.get("N", 0)),
            P=int(pi.get("P", 0)),
            C=float(pi.get("C", 0.0)),
            P_num=int(fc.get("P_num", 0)),
            P_cat=int(fc.get("P_cat", 0)),
            I_out=float(db.get("I_out", {}).get("value", 1.0)),
            I_pop=float(db.get("I_pop", {}).get("value", 1.0)),
            domain=xi.domain,
            protected_attr=xi.protected_col,
            outcome_col=xi.outcome_col,
        )
