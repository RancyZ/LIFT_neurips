"""
Data preparation: imputation, stratified split, B downsampled subsets.
Matches paper Section 3.2.3 exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def prepare_dataset(
    df: pd.DataFrame,
    outcome_col: str,
    protected_col: str,
    B: int,
    minority_ratio: float,
    test_split: float,
    random_seed: int,
    id_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    """
    Returns: (D_tr, D_te, subsets)

    Pipeline:
    1. Drop id_col if present (never a feature)
    2. Feature-wise imputation (median for numeric, mode for categorical)
    3. One-hot encode categorical feature columns so all models receive numeric input.
       The original protected_col string values are re-inserted after encoding so
       lifecycle fairness stages can still group by protected attribute.
    4. Stratified split on (outcome, protected) jointly — test_split=0.5
    5. Downsample D_tr into B subsets:
       - minority:majority ratio = minority_ratio
       - each subset drawn with replacement from majority class
       - minority class kept in full
    """
    df = df.copy()
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])
    df = impute(df, protected_col=protected_col)
    df = encode_categorical(df, outcome_col, protected_col)
    D_tr, D_te = stratified_split(df, outcome_col, protected_col, test_split, random_seed)
    subsets = make_subsets(D_tr, outcome_col, B, minority_ratio, random_seed)
    return D_tr, D_te, subsets


def encode_categorical(
    df: pd.DataFrame,
    outcome_col: str,
    protected_col: str,
) -> pd.DataFrame:
    """
    One-hot encode all object-dtype columns (excluding the outcome column).

    The protected attribute is encoded for model input (so models get numeric
    features) but the original string values are re-inserted under the same
    column name so lifecycle fairness stages can still group by them.
    """
    # Columns with string/object dtype that need encoding (not outcome)
    cat_cols = [
        c for c in df.columns
        if c != outcome_col and df[c].dtype == object
    ]
    if not cat_cols:
        return df

    # Save original protected values before they get replaced by dummies
    protected_orig = df[protected_col].copy() if protected_col in cat_cols else None

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Re-insert original protected column for fairness grouping
    if protected_orig is not None:
        df[protected_col] = protected_orig.values

    return df


def impute(df: pd.DataFrame, protected_col: str | None = None) -> pd.DataFrame:
    """
    Impute missing values using within-group probability sampling.

    For each column that has missing values, and for each subgroup defined by
    protected_col, missing values are filled by sampling from the observed value
    distribution in that group (matching the T1D paper methodology).

    Falls back to the global distribution when a group has no observed values
    for a column, and to median/mode when the column is entirely missing.
    If protected_col is None or absent, global median/mode is used throughout.
    """
    columns_with_missing = [c for c in df.columns if df[c].isnull().any()]
    if not columns_with_missing:
        return df

    # No group-aware imputation possible — use global fallback
    if protected_col is None or protected_col not in df.columns:
        return _impute_global(df)

    groups = df[protected_col].dropna().unique()
    rng = np.random.RandomState(0)

    for col in columns_with_missing:
        if col == protected_col:
            continue  # handled separately below

        for group in groups:
            mask_group   = df[protected_col] == group
            mask_missing = df[col].isnull()
            target_mask  = mask_group & mask_missing
            missing_count = int(target_mask.sum())
            if missing_count == 0:
                continue

            # Observed values within this group
            observed = df.loc[mask_group & ~mask_missing, col]
            if observed.empty:
                # Group has no observed values — fall back to global distribution
                observed = df.loc[~mask_missing, col]
            if observed.empty:
                continue

            counts = observed.value_counts(normalize=True)
            imputed = rng.choice(counts.index, size=missing_count, p=counts.values)
            df.loc[target_mask, col] = imputed

    # Impute the protected column itself using its global distribution
    if df[protected_col].isnull().any():
        observed = df[protected_col].dropna()
        if not observed.empty:
            counts = observed.value_counts(normalize=True)
            n = int(df[protected_col].isnull().sum())
            df.loc[df[protected_col].isnull(), protected_col] = rng.choice(
                counts.index, size=n, p=counts.values
            )

    return df


def _impute_global(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: numeric → median, categorical → mode."""
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
    return df


def stratified_split(
    df: pd.DataFrame,
    outcome_col: str,
    protected_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified on cross-product of (y, a)."""
    # Create a combined strata column
    strata = df[outcome_col].astype(str) + "_" + df[protected_col].astype(str)

    # Some strata groups may be too small for stratified split;
    # fall back to non-stratified if needed.
    counts = strata.value_counts()
    min_count = counts.min()
    use_stratify = min_count >= 2

    D_tr, D_te = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strata if use_stratify else None,
    )
    return D_tr.reset_index(drop=True), D_te.reset_index(drop=True)


def make_subsets(
    D_tr: pd.DataFrame,
    outcome_col: str,
    B: int,
    minority_ratio: float,
    seed: int,
) -> List[pd.DataFrame]:
    """
    Generates B downsampled training subsets.
    - minority:majority ratio = minority_ratio
    - each subset drawn with replacement from majority class
    - minority class kept in full
    """
    rng = np.random.RandomState(seed)

    counts = D_tr[outcome_col].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    minority_df = D_tr[D_tr[outcome_col] == minority_label]
    majority_df = D_tr[D_tr[outcome_col] == majority_label]

    n_minority = len(minority_df)
    # minority_ratio = n_minority / n_majority_sample  →  n_majority_sample = n_minority / minority_ratio
    n_majority_sample = max(1, int(round(n_minority / minority_ratio)))

    subsets: List[pd.DataFrame] = []
    for _ in range(B):
        maj_sample = majority_df.sample(
            n=min(n_majority_sample, len(majority_df)),
            replace=True,
            random_state=rng.randint(0, 2**31 - 1),
        )
        subset = pd.concat([minority_df, maj_sample]).sample(
            frac=1, random_state=rng.randint(0, 2**31 - 1)
        ).reset_index(drop=True)
        subsets.append(subset)

    return subsets
