"""
LIFTPipeline — top-level orchestration class.
Wires agents, preprocessing, and lifecycle runner together.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from lift.agents.data_profiler import DataProfiler
from lift.agents.governance_reporter import GovernanceReporter
from lift.agents.model_orchestrator import ModelOrchestrator
from lift.lifecycle.runner import run_lifecycle_stages
from lift.models.preprocessing import prepare_dataset
from lift.schemas import (
    DatasetContext,
    EvalResults,
    GovReport,
    LLMConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LIFTPipeline:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        llm_cfg = LLMConfig(**self.config["llm"])

        self._profiler = DataProfiler(llm_cfg)
        self._orchestrator = ModelOrchestrator(llm_cfg)
        self._reporter = GovernanceReporter(llm_cfg)

        self.last_eval_results: EvalResults = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        xi: DatasetContext,
        _progress_fn=None,
        _stop_check=None,
    ) -> GovReport:
        """
        Full pipeline execution.
        1. Profile dataset → P_D
        2. Orchestrate → (models, stages)
        3. Preprocess → (D_tr, D_te, subsets[B])
        4. Run lifecycle stages → E_D
        5. Generate report → R_D

        _progress_fn: optional callable(event: dict) for real-time progress.
        Side effects: saves outputs to outputs/ directory.
        """
        def emit(event: dict) -> None:
            if _progress_fn:
                _progress_fn(event)

        def check_stop() -> None:
            if _stop_check and _stop_check():
                raise InterruptedError("Run stopped by user.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_tag = _safe_tag(xi.domain)

        logger.info("=== LIFT Pipeline starting — dataset: %s ===", xi.domain)

        # Step 1: Profile
        logger.info("Step 1: Profiling dataset …")
        emit({"stage": "profiling", "status": "running"})
        profile = self._profiler.profile(df, xi)
        logger.info("  P_D: N=%d, P=%d, I_out=%.3f, I_pop=%.3f",
                    profile.N, profile.P, profile.I_out, profile.I_pop)
        emit({"stage": "profiling", "status": "done", "data": {
            "N": profile.N, "P": profile.P, "C": round(profile.C, 4),
            "I_out": round(profile.I_out, 3), "I_pop": round(profile.I_pop, 3),
            "P_num": profile.P_num, "P_cat": profile.P_cat,
        }})

        check_stop()

        # Step 2: Orchestrate
        logger.info("Step 2: Orchestrating model selection …")
        emit({"stage": "orchestrating", "status": "running"})
        decision = self._orchestrator.decide(profile)
        model_ids = decision.selected_model_ids()
        logger.info("  Selected models: %s", model_ids)
        logger.info("  Activated stages: %s", decision.globally_activated_stages())
        emit({"stage": "orchestrating", "status": "done", "data": {
            "models": model_ids,
            "stages": decision.globally_activated_stages(),
        }})

        check_stop()

        # Step 3: Preprocess
        logger.info("Step 3: Preprocessing (B=%d subsets) …",
                    self.config["pipeline"]["B"])
        emit({"stage": "preprocessing", "status": "running"})
        D_tr, D_te, subsets = prepare_dataset(
            df=df,
            outcome_col=xi.outcome_col,
            protected_col=xi.protected_col,
            B=self.config["pipeline"]["B"],
            minority_ratio=self.config["pipeline"]["minority_ratio"],
            test_split=self.config["pipeline"]["test_split"],
            random_seed=self.config["pipeline"]["random_seed"],
            id_col=xi.id_col,
        )
        emit({"stage": "preprocessing", "status": "done", "data": {
            "B": len(subsets),
            "D_tr": len(D_tr),
            "D_te": len(D_te),
        }})

        check_stop()

        # Step 4: Run lifecycle stages
        logger.info("Step 4: Running lifecycle stages …")
        emit({"stage": "lifecycle", "status": "running", "data": {
            "activated_stages": decision.globally_activated_stages(),
            "models": model_ids,
        }})
        run_config = dict(self.config)
        run_config["_outcome_col"] = xi.outcome_col
        run_config["_protected_col"] = xi.protected_col

        eval_results = run_lifecycle_stages(
            D_te=D_te,
            subsets=subsets,
            decision=decision,
            config=run_config,
            stop_check=_stop_check,
        )
        self.last_eval_results = eval_results
        scoreboard = [
            {
                "model_id": mid,
                "stage_id": sid,
                "score": round(r.score, 4),
                "acc": round(r.raw["Acc_i"], 4) if "Acc_i" in r.raw else None,
                "spec": round(r.raw["Spec_i"], 4) if "Spec_i" in r.raw else None,
            }
            for (mid, sid), r in eval_results.items()
        ]
        emit({"stage": "lifecycle", "status": "done", "data": {"scoreboard": scoreboard}})

        check_stop()

        # Step 5: Generate governance report
        logger.info("Step 5: Generating governance report …")
        emit({"stage": "reporting", "status": "running"})
        report = self._reporter.report(eval_results, profile, decision, xi)
        logger.info("  Primary model: %s", report.primary_model)
        emit({"stage": "reporting", "status": "done"})

        # Signal pipeline complete (exclude raw_llm_output to keep payload small)
        f = decision.dataset_flags
        emit({"stage": "pipeline_done", "status": "done", "report": {
            "primary_model": report.primary_model,
            "primary_rationale": report.primary_rationale,
            "alternatives": report.alternatives,
            "avoid": report.avoid,
            "improvement_actions": report.improvement_actions,
            "stage_interpretations": report.stage_interpretations,
            "activated_stages": decision.globally_activated_stages(),
            "dataset_flags": {
                "high_dimensionality":       f.high_dimensionality,
                "large_n":                   f.large_n,
                "high_incompleteness":       f.high_incompleteness,
                "high_outcome_imbalance":    f.high_outcome_imbalance,
                "high_population_imbalance": f.high_population_imbalance,
                "balanced_dataset":          f.balanced_dataset,
                "mixed_features":            f.mixed_features,
                "dim_ratio_threshold":       f.dim_ratio_threshold,
                "large_n_threshold":         f.large_n_threshold,
                "incompleteness_threshold":  f.incompleteness_threshold,
                "imbalance_threshold":       f.imbalance_threshold,
                "balanced_threshold":        f.balanced_threshold,
            },
        }})

        # Step 6: Save outputs
        logger.info("Step 6: Saving outputs …")
        self._save_outputs(profile, eval_results, report, dataset_tag, ts)

        logger.info("=== LIFT Pipeline complete ===")
        return report

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    def _save_outputs(self, profile, eval_results, report, tag, ts) -> None:
        base = Path(os.path.dirname(__file__)).parent / "outputs"

        # Profile JSON
        profile_path = base / "profiles" / f"{tag}_{ts}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile.__dict__, indent=2))

        # Scoreboard CSV
        sb_path = base / "scoreboards" / f"{tag}_{ts}.csv"
        sb_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_id": mid,
                "stage_id": sid,
                "score": r.score,
                **{f"raw_{k}": v for k, v in r.raw.items() if not isinstance(v, list)},
            }
            for (mid, sid), r in eval_results.items()
        ]
        import csv
        if rows:
            with open(sb_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        # Report JSON
        rp_json = base / "reports" / f"{tag}_{ts}.json"
        rp_json.parent.mkdir(parents=True, exist_ok=True)
        rp_json.write_text(json.dumps(report.__dict__, indent=2))

        # Report Markdown
        rp_md = base / "reports" / f"{tag}_{ts}.md"
        rp_md.write_text(_report_to_md(report))

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        # Try absolute path, then relative to this file
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(p) as f:
            return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_tag(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:40]


def _report_to_md(report: GovReport) -> str:
    lines = [
        "# LIFT Governance Report",
        "",
        f"## Primary Model: `{report.primary_model}`",
        "",
        report.primary_rationale,
        "",
        "## Alternative Models",
        "",
    ]
    for alt in report.alternatives:
        lines.append(f"- **{alt.get('model', '')}**: {alt.get('rationale', '')}")
    lines += ["", "## Models to Avoid", ""]
    for av in report.avoid:
        lines.append(f"- **{av.get('model', '')}**: {av.get('reason', '')}")
    lines += ["", "## Improvement Actions", ""]
    for model, actions in report.improvement_actions.items():
        lines.append(f"### {model}")
        for action in actions:
            lines.append(f"- {action}")
    lines += ["", "## Stage Interpretations", ""]
    for stage, narrative in report.stage_interpretations.items():
        lines.append(f"### {stage}")
        lines.append(narrative)
    return "\n".join(lines) + "\n"
