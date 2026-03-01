"""
DataProfiler — Agent 1 (Φ_data).

Paper reference: Section 3.1.1, Equation 1.
  P_D ~ Φ_data(g(D, ξ_data))

Design (per paper):
- All numeric metrics are computed DETERMINISTICALLY from the DataFrame.
- The LLM adds a clinical narrative interpreting those metrics in context.
- DataProfile is built from deterministic stats, not from LLM output,
  to ensure correctness and auditability.

Metrics (Section 3.2.1):
  C     = 1 - (1/NP) Σ_i Σ_j m_ij      (incompleteness over P feature cols only)
  I_out = max_o(N_o) / min_o(N_o)       (outcome imbalance ratio)
  I_pop = max_a(N_a) / min_a(N_a)       (population group imbalance ratio)
  T     = {P_num, P_cat}                (feature type composition)
  S     = N × P                         (dataset scale; stored as N and P separately)
"""
from __future__ import annotations

import logging

import pandas as pd

from lift.agents.base_agent import BaseAgent
from lift.prompts import data_profiler_prompt as dp_prompt
from lift.schemas import DataProfile, DatasetContext, LLMConfig

logger = logging.getLogger(__name__)


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
        Step 1: Compute all metrics deterministically from df.
        Step 2: Call LLM to produce a clinical narrative (not to recompute metrics).
        Step 3: Build and return DataProfile from deterministic stats + LLM narrative.
        """
        stats = self._compute_stats(df, xi)

        narrative = ""
        try:
            user_prompt = dp_prompt.build_user_prompt(stats, xi)
            result = self.call_llm(
                prompt=user_prompt,
                system_prompt=dp_prompt.SYSTEM,
                expect_json=True,
            )
            narrative = str(result.get("profile_narrative", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DataProfiler LLM narrative failed (non-fatal): %s", exc)

        return DataProfile(
            N=stats["N"],
            P=stats["P"],
            C=stats["C"],
            P_num=stats["P_num"],
            P_cat=stats["P_cat"],
            I_out=stats["I_out"],
            I_pop=stats["I_pop"],
            domain=xi.domain,
            protected_attr=xi.protected_col,
            outcome_col=xi.outcome_col,
            profile_narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Deterministic statistics  (Section 3.2.1)
    # ------------------------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame, xi: DatasetContext) -> dict:
        """
        Returns: { N, P, C, I_out, I_pop, P_num, P_cat }

        C = 1 - (1/NP) Σ_i Σ_j m_ij
            Computed over the P predictive feature columns only,
            excluding outcome (y) and protected attribute (a).

        I_out = max(class_counts) / min(class_counts)
        I_pop = max(group_counts) / min(group_counts)
        """
        # P = predictive features, excluding y and a  (paper definition)
        feature_cols = [
            c for c in df.columns
            if c not in (xi.outcome_col, xi.protected_col)
        ]

        N = len(df)
        P = len(feature_cols)

        # Incompleteness: fraction of missing values over feature cells only
        total_cells = N * P
        observed_cells = int(df[feature_cols].notna().sum().sum()) if P > 0 else 0
        C = round(1.0 - observed_cells / total_cells, 6) if total_cells > 0 else 0.0

        # Imbalance ratios
        I_out = self._imbalance(df[xi.outcome_col])
        I_pop = self._imbalance(df[xi.protected_col])

        # Feature type composition  T = {P_num, P_cat}
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

    @staticmethod
    def _imbalance(series: pd.Series) -> float:
        """max_count / min_count across all values in series."""
        counts = series.dropna().value_counts()
        if len(counts) < 2:
            return 1.0
        return round(float(counts.max()) / float(counts.min()), 4)
