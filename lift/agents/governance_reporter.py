"""
GovernanceReporter — Agent 3 (Φ_rep).
Synthesizes lifecycle results into a governance report (R_D).
"""
from __future__ import annotations

import io

from lift.agents.base_agent import BaseAgent
from lift.prompts import reporter_prompt as rp_prompt
from lift.schemas import (
    DataProfile,
    DatasetContext,
    EvalResults,
    GovReport,
    LLMConfig,
    OrchestratorDecision,
)


class GovernanceReporter(BaseAgent):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        eval_results: EvalResults,
        profile: DataProfile,
        decision: OrchestratorDecision,
        xi: DatasetContext,
    ) -> GovReport:
        """
        Formats all lifecycle scores into a structured prompt.
        Calls LLM for synthesis and governance recommendations.
        Does NOT recompute any metric.
        """
        scores_table = self._format_scores_for_prompt(eval_results)
        user_prompt = rp_prompt.build_user_prompt(scores_table, profile, xi)
        raw_text = self.call_llm(
            prompt=user_prompt,
            system_prompt=rp_prompt.SYSTEM,
            expect_json=False,  # capture raw text too
        )
        # Parse the JSON from the raw text
        parsed = self._parse_json(raw_text)
        return self._build_report(parsed, raw_text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_scores_for_prompt(self, eval_results: EvalResults) -> str:
        """Renders normalised stage scores as a markdown table for LLM context."""
        if not eval_results:
            return "(no evaluation results)"

        # Collect all model ids and stage ids
        model_ids: list[str] = []
        stage_ids: list[str] = []
        for (mid, sid) in eval_results:
            if mid not in model_ids:
                model_ids.append(mid)
            if sid not in stage_ids:
                stage_ids.append(sid)

        model_ids.sort()
        stage_ids.sort()

        buf = io.StringIO()
        header = "| Model | " + " | ".join(stage_ids) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(stage_ids)) + "|"
        buf.write(header + "\n")
        buf.write(sep + "\n")
        for mid in model_ids:
            row_vals = []
            for sid in stage_ids:
                result = eval_results.get((mid, sid))
                row_vals.append(f"{result.score:.3f}" if result else "  N/A  ")
            buf.write(f"| {mid} | " + " | ".join(row_vals) + " |\n")
        return buf.getvalue()

    def _build_report(self, parsed: dict, raw_text: str) -> GovReport:
        return GovReport(
            primary_model=parsed.get("primary_model", ""),
            primary_rationale=parsed.get("primary_rationale", ""),
            alternatives=parsed.get("alternatives", []),
            avoid=parsed.get("avoid", []),
            improvement_actions=parsed.get("improvement_actions", {}),
            stage_interpretations=parsed.get("stage_interpretations", {}),
            raw_llm_output=raw_text,
        )
