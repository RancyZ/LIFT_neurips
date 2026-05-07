"""
lift/lift_scoreboards.py
Scoreboard computation and PDF radar-chart export for the LIFT framework.
Implements the five-axis scoreboard from Section 3.2 of the LIFT paper.

Axis definitions
----------------
0. Risk Prediction Models  — LLM orchestrator composite rubric score
1. Generalizability        — e_gen = 0.5·Acc_i + 0.5·Spec_i, cross-model normalised
2. Deployment              — e_dep = (S_rob + S_par + S_exp) / 3, cross-model normalised
3. Monitoring & Updating   — inverted Wasserstein drift, cross-model normalised
4. Learning & Optimization — inverted mean train time, cross-model normalised

All lifecycle scores arrive here already normalised to [0, 1] by
`_normalise_scores` in lifecycle/runner.py. This module re-packages them
into human-readable axes and renders per-model PDF radar charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Axis metadata
# ---------------------------------------------------------------------------

STAGE_ID_TO_LABEL: Dict[str, str] = {
    "l1_learning_opt":     "Learning & Optimization",
    "l2_generalizability": "Generalizability",
    "l3_deployment":       "Deployment",
    "l4_monitoring":       "Monitoring & Updating",
}

# Canonical display order — "Risk Prediction Models" always at top (12 o'clock)
AXIS_ORDER: List[str] = [
    "Risk Prediction Models",
    "Generalizability",
    "Deployment",
    "Monitoring & Updating",
    "Learning & Optimization",
]

# Per-model colours (matches plot_scoreboard.py)
MODEL_COLORS: Dict[str, str] = {
    "LR":  "#1f77b4",
    "BR":  "#ff7f0e",
    "SVM": "#d62728",
    "RF":  "#8c564b",
    "VAE": "#17becf",
}
_FALLBACK_COLORS = ["#2ca02c", "#9467bd", "#e377c2", "#7f7f7f", "#bcbd22", "#aec7e8"]

# Per-label display tuning (matches plot_scoreboard.py)
_LABEL_RADIUS: Dict[str, float] = {
    "Risk Prediction Models":  1.30,
    "Learning & Optimization": 1.10,
    "Generalizability":        1.20,
    "Deployment":              1.10,
    "Monitoring & Updating":   1.10,
}
_LABEL_ANGLE_SHIFT: Dict[str, float] = {
    "Risk Prediction Models":  -0.50,
    "Learning & Optimization":  0.00,
    "Generalizability":        -0.40,
    "Deployment":               0.00,
    "Monitoring & Updating":    0.00,
}
_LABEL_ROTATION: Dict[str, float] = {
    "Risk Prediction Models":    0,
    "Learning & Optimization": 270,
    "Generalizability":          0,
    "Deployment":               90,
    "Monitoring & Updating":    90,
}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scoreboards(
    eval_results,   # EvalResults: Dict[Tuple[str, str], StageResult]
    decision,       # OrchestratorDecision
) -> Dict[str, Dict[str, float]]:
    """
    Assemble per-model scoreboards from already-normalised EvalResults.

    Parameters
    ----------
    eval_results : EvalResults
        ``Dict[(model_id, stage_id)] → StageResult`` where every
        ``StageResult.score`` is already normalised to [0, 1] by the runner.
    decision : OrchestratorDecision
        Full orchestrator output; ``model_routes[i].model_score.composite_score``
        supplies the Risk Prediction Models axis value.

    Returns
    -------
    Dict[model_id, Dict[axis_label, float]]
        e.g. ``{"LR": {"Risk Prediction Models": 0.82, "Generalizability": 0.71, …}}``
    """
    orch_scores: Dict[str, float] = {
        route.model_id: round(route.model_score.composite_score, 6)
        for route in decision.model_routes
    }

    all_model_ids = sorted({mid for (mid, _) in eval_results})

    scoreboard: Dict[str, Dict[str, float]] = {}
    for model_id in all_model_ids:
        axes: Dict[str, float] = {}

        axes["Risk Prediction Models"] = orch_scores.get(model_id, 0.5)

        key = (model_id, "l2_generalizability")
        if key in eval_results:
            axes["Generalizability"] = round(eval_results[key].score, 6)

        key = (model_id, "l3_deployment")
        if key in eval_results:
            axes["Deployment"] = round(eval_results[key].score, 6)

        key = (model_id, "l4_monitoring")
        if key in eval_results:
            axes["Monitoring & Updating"] = round(eval_results[key].score, 6)

        key = (model_id, "l1_learning_opt")
        if key in eval_results:
            axes["Learning & Optimization"] = round(eval_results[key].score, 6)

        scoreboard[model_id] = axes

    return scoreboard


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_scores_to_pdf(
    scoreboard_dict: Dict[str, Dict[str, float]],
    activated_stages: List[str],
    output_dir: str | Path,
) -> None:
    """
    Render per-model radar charts as PDF files (matches plot_scoreboard.py style).

    Outputs
    -------
    <output_dir>/lifecycle_radar_<model>.pdf  — one file per model
    <output_dir>/lifecycle_radar_all.pdf      — combined multi-page PDF

    Parameters
    ----------
    scoreboard_dict : Dict[model_id, Dict[axis_label, float]]
        Output of :func:`compute_scoreboards`.
    activated_stages : list[str]
        Human-readable stage labels to show as radar axes.
        ``"Risk Prediction Models"`` is prepended automatically if absent.
    output_dir : str | Path
        Directory where PDFs are written (created if needed).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build ordered axis list
    stages = list(activated_stages)
    if "Risk Prediction Models" not in stages:
        stages.insert(0, "Risk Prediction Models")
    ordered = [s for s in AXIS_ORDER if s in stages]
    extras  = [s for s in stages if s not in AXIS_ORDER]
    metrics = ordered + extras

    num_vars      = len(metrics)
    angles        = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    tick_vals     = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Assign colours — known models use fixed palette, unknowns get fallbacks
    models = list(scoreboard_dict.keys())
    fallback_iter = iter(_FALLBACK_COLORS)
    model_colors  = {m: MODEL_COLORS.get(m, next(fallback_iter, "#333333")) for m in models}

    out_pdf_all = out_dir / "lifecycle_radar_all.pdf"
    with PdfPages(out_pdf_all) as pdf:
        for model in models:
            color  = model_colors[model]
            scores = scoreboard_dict[model]
            vals   = [float(np.clip(scores.get(m, 0.0), 0, 1)) for m in metrics]
            vals_closed = vals + vals[:1]

            fig = plt.figure(figsize=(25, 8.5))
            ax  = plt.subplot(111, polar=True)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels([])
            ax.set_ylim(0, 1.0)
            ax.set_yticks(tick_vals)
            ax.set_yticklabels([])

            ax.yaxis.grid(True, linestyle="--", linewidth=1.0, alpha=0.5,  color="black")
            ax.xaxis.grid(True, linestyle="-",  linewidth=0.8, alpha=0.35, color="black")
            ax.spines["polar"].set_linestyle("--")
            ax.spines["polar"].set_linewidth(1.0)
            ax.spines["polar"].set_alpha(0.5)
            ax.spines["polar"].set_color("black")

            # Axis labels
            for i, name in enumerate(metrics):
                base_angle = angles[i]
                angle      = base_angle + _LABEL_ANGLE_SHIFT.get(name, 0.0)
                radius     = _LABEL_RADIUS.get(name, 1.22)
                rotation   = _LABEL_ROTATION.get(name, np.degrees(base_angle))

                if 90 < rotation < 270:
                    rotation += 180

                if np.pi / 2 < base_angle < 3 * np.pi / 2:
                    ha = "right"
                elif np.isclose(base_angle, np.pi / 2) or np.isclose(base_angle, 3 * np.pi / 2):
                    ha = "center"
                else:
                    ha = "left"

                ax.text(
                    angle, radius, name,
                    ha=ha, va="center", fontsize=35,
                    rotation=rotation, rotation_mode="anchor",
                    clip_on=False,
                )

            # Model polygon
            ax.plot(angles_closed, vals_closed, linewidth=2.2, color=color, alpha=0.95)
            ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)

            # Tick value labels along each axis
            _draw_metric_ticks(ax, angles, tick_vals, fontsize=26, color="black")

            plt.tight_layout()

            out_pdf = out_dir / f"lifecycle_radar_{model}.pdf"
            plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _draw_metric_ticks(
    ax,
    angles: list,
    tick_vals: list,
    fontsize: int = 26,
    color: str = "black",
) -> None:
    """Draw numeric tick labels along each radar axis."""
    import numpy as np
    for angle in angles:
        for t in tick_vals:
            if np.isclose(t, 0.0):
                continue
            r   = float(np.clip(t, 0, 1))
            txt = "1" if np.isclose(t, 1.0) else f"{t:.2f}".rstrip("0").rstrip(".")
            ax.text(angle, r, txt, ha="center", va="center", fontsize=fontsize, color=color)
