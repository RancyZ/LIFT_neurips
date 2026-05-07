"""
GovernanceReporter — Agent 3 (Φ_rep).

Paper reference: Section 3.1.3, Equation 3.
  R_D ~ Φ_rep(g(E_D, ξ_rep))
  R_D = ({e_l_j}_{j∈J_D}, γ_D)

The reporter synthesises E_D (all lifecycle scores + raw metrics) into:
  {e_l_j} — per-stage interpretations
  γ_D     — model recommendations and improvement actions

It does NOT recompute any metric; it interprets and contextualises.
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
        Formats lifecycle scores + raw metrics into a structured prompt.
        Passes dataset flags, orchestrator summary, and profile narrative
        so the LLM can ground recommendations in dataset-specific risks.
        Calls LLM for synthesis. Does NOT recompute any metric.
        """
        scores_table = self._format_scores_table(eval_results)
        raw_metrics_block = self._format_raw_metrics(eval_results)
        actual_stages = sorted({sid for (_, sid) in eval_results})
        user_prompt = rp_prompt.build_user_prompt(
            scores_table=scores_table,
            raw_metrics_block=raw_metrics_block,
            profile=profile,
            decision=decision,
            xi=xi,
            actual_stages=actual_stages,
        )
        raw_text = self.call_llm(
            prompt=user_prompt,
            system_prompt=rp_prompt.SYSTEM,
            expect_json=False,  # capture raw text for raw_llm_output field
        )
        parsed = self._parse_json(raw_text)
        return self._build_report(parsed, raw_text)

    # ------------------------------------------------------------------
    # Prompt formatting helpers
    # ------------------------------------------------------------------

    def _format_scores_table(self, eval_results: EvalResults) -> str:
        """Normalised [0,1] score matrix — model rows × stage columns."""
        if not eval_results:
            return "(no evaluation results)"

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
        buf.write("| Model | " + " | ".join(stage_ids) + " |\n")
        buf.write("|-------|" + "|".join(["-------"] * len(stage_ids)) + "|\n")
        for mid in model_ids:
            row_vals = []
            for sid in stage_ids:
                result = eval_results.get((mid, sid))
                row_vals.append(f"{result.score:.3f}" if result else "N/A")
            buf.write(f"| {mid} | " + " | ".join(row_vals) + " |\n")
        return buf.getvalue()

    def _format_raw_metrics(self, eval_results: EvalResults) -> str:
        """
        Key raw metrics from StageResult.raw, grouped by model and stage.
        Gives the LLM access to underlying values (Acc, Spec, EOP gap,
        drift distance, etc.) beyond the normalised score alone.
        """
        if not eval_results:
            return "(no raw metrics)"

        lines = []
        for (mid, sid), result in sorted(eval_results.items()):
            numeric_raw = {
                k: round(float(v), 4)
                for k, v in result.raw.items()
                if isinstance(v, (int, float))
            }
            if numeric_raw:
                kv = ", ".join(f"{k}={v}" for k, v in numeric_raw.items())
                lines.append(f"  {mid} / {sid}: {kv}")
        return "\n".join(lines) if lines else "(no raw metrics available)"

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

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
