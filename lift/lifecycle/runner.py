"""
Lifecycle runner — dispatches activated stages per selected model.
Parallelises across models using concurrent.futures.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd

from lift.schemas import EvalResults, OrchestratorDecision, StageResult

logger = logging.getLogger(__name__)

# The five models always shown in the frontend — all must be evaluated.
FIXED_MODEL_IDS = ["LR", "BR", "SVM", "RF", "VAE"]


def run_lifecycle_stages(
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
    outcome_col = config["_outcome_col"]
    protected_col = config.get("_protected_col", None)

    eval_results: EvalResults = {}

    # l4 monitoring is model-agnostic — always run (Wasserstein drift is always meaningful)
    from lift.lifecycle.monitoring import evaluate_monitoring
    monitoring_result = evaluate_monitoring(
        subsets=subsets,
        D_te=D_te,
        outcome_col=outcome_col,
        protected_col=protected_col,
        config=config,
    )

    def _run_model(model_id: str) -> dict:
        results = {}
        try:
            # ── Pre-compute best hyperparams once per subset ──────────────────
            from lift.models.tuner import tune_hyperparams
            hp_grid      = config.get("models", {}).get("hyperparams", {}).get(model_id, {})
            cv_folds_cfg = config.get("pipeline", {}).get("cv_folds", 5)
            seed         = config.get("pipeline", {}).get("random_seed", 42)

            best_params_per_b: List[dict] = []
            for b_idx, subset in enumerate(subsets):
                if len(subset) < 2:
                    best_params_per_b.append({})
                    continue
                drop = [c for c in [outcome_col, protected_col] if c and c in subset.columns]
                X_b  = subset.drop(columns=drop).select_dtypes(include="number")
                y_b  = subset[outcome_col]
                bp   = tune_hyperparams(
                    model_id=model_id,
                    X_tr=X_b, y_tr=y_b,
                    param_grid=hp_grid,
                    cv_folds=min(cv_folds_cfg, len(subset) // 2),
                    random_seed=seed + b_idx,
                )
                best_params_per_b.append(bp)

            # ── All stages run for every model ────────────────────────────────
            from lift.lifecycle.learning_opt import evaluate_learning_opt
            results[(model_id, "l1_learning_opt")] = evaluate_learning_opt(
                model_id, subsets, D_te, outcome_col, protected_col, best_params_per_b, config
            )

            from lift.lifecycle.generalizability import evaluate_generalizability
            results[(model_id, "l2_generalizability")] = evaluate_generalizability(
                model_id, subsets, D_te, outcome_col, protected_col, best_params_per_b, config
            )

            from lift.lifecycle.deployment import evaluate_deployment
            results[(model_id, "l3_deployment")] = evaluate_deployment(
                model_id, subsets, D_te, outcome_col, protected_col, best_params_per_b, config
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Model '%s' failed — error: %s: %s",
                model_id, type(exc).__name__, exc,
                exc_info=True,
            )
        return results

    # Parallelise across all 5 fixed models
    with ThreadPoolExecutor(max_workers=min(4, len(FIXED_MODEL_IDS))) as executor:
        futures = {executor.submit(_run_model, mid): mid for mid in FIXED_MODEL_IDS}
        for future in as_completed(futures):
            if stop_check and stop_check():
                raise InterruptedError("Run stopped by user.")
            model_result = future.result()
            eval_results.update(model_result)

    # Attach l4 monitoring results for all 5 models
    for mid in FIXED_MODEL_IDS:
        eval_results[(mid, "l4_monitoring")] = StageResult(
            stage_id="l4_monitoring",
            model_id=mid,
            score=monitoring_result.score,
            raw=monitoring_result.raw,
        )

    # Normalise scores within each stage — derive stages from actual results
    actual_stages = sorted({sid for (_, sid) in eval_results})
    eval_results = _normalise_scores(eval_results, actual_stages)
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
