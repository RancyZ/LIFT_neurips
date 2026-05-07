# LIFT: Agentic LLMs for Lifecycle-Based Workflow Testing in Healthcare Risk Prediction

Code for **LIFT: Agentic LLMs for Lifecycle-Based Workflow Testing in Healthcare Risk Prediction**. 

LIFT profiles a healthcare dataset, selects candidate models and evaluation stages based on its characteristics, assesses them across selected lifecycle stages, and produces a structured report with actionable recommendations.

## Requirements

- Python 3.9+
- An [OpenAI](https://platform.openai.com) API key

## Setup

```bash
git clone <repository-url>
cd LIFT
pip install -r requirements.txt
```

PyTorch, PyMC, and Bambi are included in the requirements — first install may take a few minutes.

## Running

### Web interface

```bash
python app.py
```

Open `http://localhost:8000`, upload your dataset (CSV or XLSX), fill in the column names, and paste your OpenAI API key. Results stream in real time and are saved to `outputs/`.

### Command line

```bash
python run_lift.py \
  --api-key sk-… \
  --dataset your_data.csv \
  --outcome y \
  --protected race \
  --domain "your domain here" \
  --cohort "cohort description"
```

## Dataset format

Any CSV or XLSX file with:
- A binary outcome column (0/1)
- A categorical protected attribute column (e.g. race, sex)

Column names are case-sensitive. Missing values up to ~10% are handled automatically. Pass `--id-col` (CLI) or the optional ID column field (web UI) to exclude a patient identifier from the feature set.

## Pipeline stages

| Stage | ID | Dimensions evaluated |
|-------|----|----------------------|
| Learning efficiency | L1 | Sample efficiency, learning curve behaviour |
| Generalizability | L2 | Discriminative power, missing-data robustness |
| Deployment | L3 | Robustness, subgroup parity, explainability |
| Monitoring | L4 | Drift detection |

Stages are activated adaptively per model based on dataset characteristics (sample size, dimensionality, class imbalance, missingness). The LLM orchestrator selects the candidate model set and routes each model through the relevant stages.

## Outputs

Each run saves to `outputs/`:

```
outputs/
  profiles/     — dataset statistics (N, P, missingness, class imbalance, group imbalance)
  scoreboards/  — per-model lifecycle scores
  reports/      — governance report (JSON + Markdown)
```

The governance report identifies the recommended primary model, ranked alternatives, models to avoid, and per-model improvement actions.

## Configuration

Model and pipeline settings are in `lift/config.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `llm.model` | `gpt-5.2` | Any model name from the [OpenAI models](https://platform.openai.com/docs/models) list |
| `pipeline.test_split` | `0.5` | Train/test split ratio |
| `pipeline.cv_folds` | `5` | Cross-validation folds for hyperparameter tuning |
| `evaluation.shap_top_H` | `20` | Number of top features used in explainability score |
| `models.candidate_set` | `[LR, BR, KNN, SVM, DT, RF, MLP, RN, VAE]` | Models considered by the orchestrator |

## Project structure

```
lift/
  agents/       — LLM agents (data profiler, model orchestrator, governance reporter)
  evaluation/   — fairness, explainability, and metrics modules
  lifecycle/    — L1–L4 stage runners
  models/       — model factory, preprocessing, hyperparameter tuner
  prompts/      — prompt templates for each agent
  pipeline.py   — top-level pipeline orchestration
  schemas.py    — shared dataclasses (DataProfile, GovReport, etc.)
  config.yaml   — pipeline and model configuration
app.py          — FastAPI web server with SSE streaming
run_lift.py     — CLI entry point
```
