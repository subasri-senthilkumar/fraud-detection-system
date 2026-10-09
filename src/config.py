"""Central configuration for the fraud detection system.

Everything that more than one module needs (column names, risk thresholds,
paths, seeds) lives here so that training and inference can never drift apart.
Runtime values that differ per environment are read from environment variables.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_RAW_DATA = DATA_RAW_DIR / "transactions.csv"

MODEL_FILENAME = "fraud_model.pkl"
PREPROCESSOR_FILENAME = "preprocessing_pipeline.pkl"
METADATA_FILENAME = "model_metadata.json"


def get_model_dir() -> Path:
    """Directory holding model artifacts (read at call time so tests can override)."""
    return Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))


# --------------------------------------------------------------------------- #
# Reproducibility / split
# --------------------------------------------------------------------------- #
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = 0.2
TARGET = "is_fraud"  # 0 = legitimate, 1 = fraudulent
ID_COLUMNS = ["transaction_id", "customer_id"]

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# Categorical levels are fixed up-front. This (a) gives the API a single source
# of truth for validation, (b) makes the one-hot width deterministic, which lets
# us tell SMOTENC exactly which columns are categorical, and (c) means an unseen
# level at inference time becomes an all-zero vector instead of a crash.
CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "merchant_category": [
        "grocery", "restaurant", "fuel", "travel", "electronics",
        "clothing", "entertainment", "health", "online_services",
    ],
    "payment_channel": ["pos", "online", "mobile_app", "atm"],
    "device_type": ["mobile", "desktop", "tablet", "pos_terminal"],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_LEVELS)

# Columns that must exist in any CSV / API payload (values may be NaN where the
# imputer can cope; see feature_engineering.py).
RAW_REQUIRED_COLUMNS = [
    "timestamp", "amount", "merchant_category", "payment_channel", "device_type",
    "is_new_device", "account_age_days", "distance_from_home_km",
    "country_mismatch", "txn_count_24h", "avg_amount_30d", "time_since_last_txn_min",
]

# Model-side feature groups (AFTER feature engineering).
NUMERIC_FEATURES = [
    "amount", "amount_log", "account_age_days", "distance_from_home_km",
    "distance_log", "txn_count_24h", "avg_amount_30d", "amount_to_avg_ratio",
    "time_since_last_txn_min", "hour",
]
BINARY_FEATURES = [
    "is_new_device", "country_mismatch", "is_night", "is_weekend",
    "is_rapid_succession", "is_high_velocity", "is_new_account",
]

# Rule thresholds used by derived features (documented in the README).
NIGHT_HOURS = range(0, 6)          # 00:00-05:59
HIGH_VELOCITY_TXN_24H = 6          # >= 6 transactions in 24h
RAPID_SUCCESSION_MINUTES = 5.0     # < 5 minutes since previous transaction
NEW_ACCOUNT_DAYS = 30              # account younger than 30 days

# --------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------- #
# ML decision: do NOT oversample to 1:1. Full balancing makes the model badly
# over-predict fraud (inflated probabilities). Oversampling the minority class to
# 20% of the majority is a milder, commonly better-calibrated choice.
SMOTE_SAMPLING_STRATEGY = float(os.getenv("SMOTE_SAMPLING_STRATEGY", "0.2"))
SMOTE_K_NEIGHBORS = 5

# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #
MODEL_VERSION_BASE = os.getenv("MODEL_VERSION_BASE", "1.0.0")


# --------------------------------------------------------------------------- #
# Risk thresholds (configurable, never hard-coded elsewhere)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskThresholds:
    """Upper bounds (inclusive) of the LOW and MEDIUM bands on the 0-100 score.

    Defaults: 0-30 LOW, 31-70 MEDIUM, 71-100 HIGH.
    """

    low_max: int = 30
    medium_max: int = 70

    def __post_init__(self) -> None:
        if not (0 <= self.low_max < self.medium_max <= 99):
            raise ValueError(
                "Risk thresholds must satisfy 0 <= RISK_LOW_MAX < RISK_MEDIUM_MAX <= 99 "
                f"(got low_max={self.low_max}, medium_max={self.medium_max})"
            )

    @classmethod
    def from_env(cls) -> "RiskThresholds":
        return cls(
            low_max=int(os.getenv("RISK_LOW_MAX", "30")),
            medium_max=int(os.getenv("RISK_MEDIUM_MAX", "70")),
        )

    def as_dict(self) -> dict[str, int]:
        return {"low_max": self.low_max, "medium_max": self.medium_max}


def get_decision_threshold_override() -> float | None:
    """Optional override of the binary fraud/legit decision threshold."""
    raw = os.getenv("DECISION_THRESHOLD")
    if not raw:
        return None
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError("DECISION_THRESHOLD must be between 0 and 1 (exclusive)")
    return value


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
