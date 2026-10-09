"""Data loading, cleaning, EDA helpers and the preprocessing pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    RAW_REQUIRED_COLUMNS,
    TARGET,
    TEST_SIZE,
)
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate the demo dataset with "
            "`python -m src.data_generation` or place your own CSV there."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required = RAW_REQUIRED_COLUMNS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    bad_target = set(df[TARGET].dropna().unique()) - {0, 1}
    if bad_target:
        raise ValueError(f"Target '{TARGET}' must contain only 0/1, found {sorted(bad_target)}")


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and rows without a label; report what was done.

    Outliers are deliberately NOT removed: in fraud detection the extreme
    values (huge amounts, bursts of transactions) are frequently the signal.
    """
    validate_schema(df)
    report: dict = {"rows_in": int(len(df))}

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n_dup

    n_unlabelled = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    report["unlabelled_removed"] = n_unlabelled

    report["missing_values"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    report["rows_out"] = int(len(df))
    logger.info("Cleaning report: %s", report)
    return df, report


def class_distribution(df: pd.DataFrame) -> dict:
    counts = df[TARGET].value_counts().to_dict()
    n_legit, n_fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    total = max(n_legit + n_fraud, 1)
    return {
        "legitimate": n_legit,
        "fraud": n_fraud,
        "fraud_rate": n_fraud / total,
        "imbalance_ratio": (n_legit / n_fraud) if n_fraud else float("inf"),
    }


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Count Tukey-fence outliers per column (reporting only - nothing is dropped)."""
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n_out, "pct_outliers": 100 * n_out / max(len(s), 1)})
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split on the RAW rows. Resampling (SMOTE) happens later and only
    inside the training folds, so the test set stays a faithful sample of reality."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_column_transformer() -> ColumnTransformer:
    """Impute + scale numerics, pass binaries, one-hot categoricals.

    Column order of the output is fixed: numeric, binary, categorical (one-hot).
    train.py relies on this order to tell SMOTENC which columns are categorical.
    Scaling is required for Logistic Regression and harmless for tree models.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # fitted on training data only
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        categories=[CATEGORICAL_LEVELS[c] for c in CATEGORICAL_FEATURES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def n_output_features() -> tuple[int, int]:
    """(number of numeric columns, total columns) produced by the column transformer."""
    n_num = len(NUMERIC_FEATURES)
    n_total = n_num + len(BINARY_FEATURES) + sum(len(v) for v in CATEGORICAL_LEVELS.values())
    return n_num, n_total


def build_preprocessing_pipeline() -> Pipeline:
    """Raw frame -> engineered frame -> model matrix (unfitted)."""
    return Pipeline([("features", FeatureEngineer()), ("prep", build_column_transformer())])
