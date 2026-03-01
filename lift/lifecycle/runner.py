"""
Lifecycle runner — dispatches activated stages per selected model.
Parallelises across models using concurrent.futures.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd

from lift.schemas import EvalResults, ModelLifecycleRoute, OrchestratorDecision, StageResult

logger = logging.getLogger(__name__)


def run_lifecycle_stages(
    D_tr: pd.DataFrame,
    D_te: pd.DataFrame,
    subsets: List[pd.DataFrame],
    decision: OrchestratorDecision,
    config: dict,
    stop_check=None,
) -> EvalResults:
    """
    Dispatches to activated stage modules for each selected model.
    Returns EvalResults dict keyed (model_id, stage_id) → StageResult.
    Parallelises across models where possible.
    """
    model_routes = decision.model_routes
    stages = decision.globally_activated_stages()

    outcome_col = config["_outcome_col"]
    protected_col = config.get("_protected_col", None)

    eval_results: EvalResults = {}

    # l4 monitoring is model-agnostic — run once
    monitoring_result: StageResult | None = None
    if "l4_monitoring" in stages:
        from lift.lifecycle.monitoring import evaluate_monitoring
        monitoring_result = evaluate_monitoring(
            subsets=subsets,
            D_te=D_te,
            outcome_col=outcome_col,
            config=config,
        )

    def _run_model(route: ModelLifecycleRoute) -> dict:
        model_id = route.model_id
        model_stages = set(route.activated_stages.keys())
        results = {}
        try:
            if "l1_learning_opt" in model_stages:
                from lift.lifecycle.learning_opt import evaluate_learning_opt
                r = evaluate_learning_opt(model_id, subsets, D_te, outcome_col, config)
                results[(model_id, "l1_learning_opt")] = r

            if "l2_generalizability" in model_stages:
                from lift.lifecycle.generalizability import evaluate_generalizability
                r = evaluate_generalizability(model_id, subsets, D_te, outcome_col, config)
                results[(model_id, "l2_generalizability")] = r

            if "l3_deployment" in model_stages:
                from lift.lifecycle.deployment import evaluate_deployment
                r = evaluate_deployment(
                    model_id, subsets, D_te, outcome_col, protected_col, config
                )
                results[(model_id, "l3_deployment")] = r

        except Exception as exc:  # noqa: BLE001
            logger.error("Error running stages for model '%s': %s", model_id, exc)
        return results

    # Parallelise across models
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(model_routes)))) as executor:
        futures = {executor.submit(_run_model, route): route.model_id for route in model_routes}
        for future in as_completed(futures):
            if stop_check and stop_check():
                raise InterruptedError("Run stopped by user.")
            model_result = future.result()
            eval_results.update(model_result)

    # Attach monitoring results for each model
    if monitoring_result is not None:
        for route in model_routes:
            stage_result = StageResult(
                stage_id="l4_monitoring",
                model_id=route.model_id,
                score=monitoring_result.score,
                raw=monitoring_result.raw,
            )
            eval_results[(route.model_id, "l4_monitoring")] = stage_result

    # Normalise scores within each stage
    eval_results = _normalise_scores(eval_results, stages)
    return eval_results


def _normalise_scores(eval_results: EvalResults, stages: List[str]) -> EvalResults:
    """
    Normalise raw scores to [0,1] within each stage across all models.

    l1 (learning_opt): lower time = better → invert then normalise
    l4 (monitoring):   lower drift = better → invert then normalise
    l2, l3: higher score = better → min-max normalise
    """
    invert_stages = {"l1_learning_opt", "l4_monitoring"}

    for stage_id in stages:
        # Collect all (key, result) for this stage
        stage_entries = {k: v for k, v in eval_results.items() if k[1] == stage_id}
        if not stage_entries:
            continue

        raw_scores = [v.score for v in stage_entries.values()]
        min_s, max_s = min(raw_scores), max(raw_scores)

        for key, result in stage_entries.items():
            if max_s == min_s:
                norm = 0.5
            elif stage_id in invert_stages:
                # Lower raw score → higher normalised score
                norm = (max_s - result.score) / (max_s - min_s)
            else:
                norm = (result.score - min_s) / (max_s - min_s)
            result.score = round(float(norm), 6)

    return eval_results
