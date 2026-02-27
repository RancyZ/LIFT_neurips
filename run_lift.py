#!/usr/bin/env python3
"""
run_lift.py — Top-level entry point script for the LIFT framework.

Usage:
    python run_lift.py
    python run_lift.py --dataset data/t1d_balanced.csv --outcome y --protected race
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the lift package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from lift.pipeline import LIFTPipeline
from lift.schemas import DatasetContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIFT Framework — run full pipeline")
    p.add_argument("--dataset", default="lift/data/t1d_balanced.csv")
    p.add_argument("--outcome", default="y")
    p.add_argument("--protected", default="race")
    p.add_argument("--domain", default="Type 1 diabetes risk prediction")
    p.add_argument("--cohort", default="TEDDY-derived SNP cohort")
    p.add_argument("--config", default="lift/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    xi = DatasetContext(
        domain=args.domain,
        outcome_col=args.outcome,
        protected_col=args.protected,
        cohort_notes=args.cohort,
    )

    pipeline = LIFTPipeline(config_path=args.config)
    report = pipeline.run(df, xi)

    print(f"\n{'='*60}")
    print(f"Primary model:  {report.primary_model}")
    print(f"Rationale:      {report.primary_rationale[:120]}…")
    print("\nImprovement actions:")
    for model, actions in report.improvement_actions.items():
        print(f"  [{model}]")
        for a in actions:
            print(f"    • {a}")
    print("="*60)


if __name__ == "__main__":
    main()
