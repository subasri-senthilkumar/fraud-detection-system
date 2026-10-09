"""Shared fixtures. A tiny model is trained once per test session (fast mode)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation import generate_transactions  # noqa: E402

GOOD_TXN = {
    "amount": 35.0, "merchant_category": "grocery", "payment_channel": "pos",
    "device_type": "pos_terminal", "is_new_device": False, "account_age_days": 900,
    "distance_from_home_km": 3.0, "country_mismatch": False, "txn_count_24h": 1,
    "avg_amount_30d": 45.0, "time_since_last_txn_min": 700.0, "timestamp": "2026-03-10T14:00:00",
}
BAD_TXN = {
    "amount": 2400.0, "merchant_category": "electronics", "payment_channel": "online",
    "device_type": "desktop", "is_new_device": True, "account_age_days": 5,
    "distance_from_home_km": 900.0, "country_mismatch": True, "txn_count_24h": 9,
    "avg_amount_30d": 60.0, "time_since_last_txn_min": 1.0, "timestamp": "2026-03-10T03:10:00",
}


@pytest.fixture(scope="session")
def small_df():
    return generate_transactions(n_rows=6000, fraud_rate=0.05, seed=7)


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, small_df):
    for module in ("xgboost", "imblearn", "shap"):
        pytest.importorskip(module)
    from src.train import run_training

    root = tmp_path_factory.mktemp("artifacts")
    run_training(small_df, output_dir=root / "models", reports_dir=root / "reports",
                 processed_dir=root / "processed", fast=True, n_iter=2, cv_folds=2,
                 explain_sample=300)
    return root / "models"


@pytest.fixture(scope="session")
def predictor(trained_dir):
    from src.predict import FraudPredictor

    return FraudPredictor.load(trained_dir)
