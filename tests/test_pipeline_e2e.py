"""
Integration test — full pipeline run on synthetic dataset.
Requires OPENAI_API_KEY or ANTHROPIC_API_KEY set in environment.
Mark as slow so it can be skipped in CI without credentials.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lift.pipeline import LIFTPipeline
from lift.schemas import DatasetContext, GovReport


def generate_synthetic_t1d(N: int = 200, P: int = 67, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic T1D-like dataset with N samples and P features."""
    rng = np.random.RandomState(seed)
    data = {}
    # Numerical features
    for i in range(P - 5):
        data[f"snp_{i}"] = rng.randn(N)
    # Categorical features
    for i in range(5):
        data[f"cat_{i}"] = rng.choice(["A", "B", "C"], size=N)
    # Protected attribute
    data["race"] = rng.choice(["White", "Black", "Hispanic", "Asian"], size=N)
    # Outcome (slightly imbalanced 60:40)
    data["y"] = rng.choice([0, 1], size=N, p=[0.6, 0.4])
    return pd.DataFrame(data)


@pytest.mark.slow
def test_e2e_t1d_balanced():
    """Full pipeline run on small synthetic T1D-like dataset."""
    import os
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        pytest.skip("No LLM API key found — skipping e2e test")

    df = generate_synthetic_t1d(N=200, P=67, seed=42)
    xi = DatasetContext(
        domain="T1D",
        outcome_col="y",
        protected_col="race",
        cohort_notes="synthetic",
    )

    # Use reduced B to keep test fast
    pipeline = LIFTPipeline(config_path="lift/config.yaml")
    pipeline.config["pipeline"]["B"] = 5  # override for speed

    report = pipeline.run(df, xi)

    assert isinstance(report, GovReport)
    valid_models = ["LR", "BR", "KNN", "SVM", "DT", "RF", "MLP", "RN", "VAE"]
    assert report.primary_model in valid_models, f"Unexpected primary model: {report.primary_model}"
    assert len(report.improvement_actions) > 0

    # All stage scores must be normalised
    for key, result in pipeline.last_eval_results.items():
        assert 0.0 <= result.score <= 1.0, (
            f"Score out of range for {key}: {result.score}"
        )
