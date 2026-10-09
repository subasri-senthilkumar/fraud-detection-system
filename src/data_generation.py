"""Synthetic transaction data generator (for demos, tests and CI).

IMPORTANT: this data is *synthetic*. It is not real banking or customer data.
It exists because public fraud datasets (e.g. the Kaggle "Credit Card Fraud
Detection" set) are anonymised PCA components with no device / location /
behavioural columns, which would make a transaction form and meaningful
explanations impossible. The generator mimics the *kinds* of signals analysts
use (amount vs. habit, night-time activity, new devices, travel distance,
velocity) with deliberate overlap between classes so the problem is not trivial.

To use a real dataset instead, map its columns onto the schema in
``src/config.py`` (RAW_REQUIRED_COLUMNS + ``is_fraud``) and save it as
``data/raw/transactions.csv`` - the rest of the pipeline is unchanged.

Usage:
    python -m src.data_generation --rows 60000 --fraud-rate 0.02
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_LEVELS,
    DEFAULT_RAW_DATA,
    RANDOM_SEED,
    TARGET,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _class_conditional_choice(rng, options, p_legit, p_fraud, is_fraud):
    legit = rng.choice(options, size=len(is_fraud), p=p_legit)
    fraud = rng.choice(options, size=len(is_fraud), p=p_fraud)
    return np.where(is_fraud, fraud, legit)


def generate_transactions(
    n_rows: int = 60_000,
    fraud_rate: float = 0.02,
    seed: int = RANDOM_SEED,
    missing_rate: float = 0.003,
    duplicate_rate: float = 0.002,
) -> pd.DataFrame:
    """Create a synthetic, imbalanced transaction table (see module docstring)."""
    rng = np.random.default_rng(seed)
    n = n_rows
    y = (rng.random(n) < fraud_rate).astype(int)
    fraud = y == 1

    # --- customers ---------------------------------------------------------
    n_customers = max(n // 15, 50)
    customer = rng.integers(0, n_customers, n)
    customer_avg = rng.lognormal(mean=np.log(60), sigma=0.7, size=n_customers)
    avg_amount = customer_avg[customer] * rng.lognormal(0, 0.1, n)

    # --- amount: fraud tends to be larger vs. habit, but 35% mimics normal ---
    mult = np.where(fraud, rng.lognormal(0.9, 0.9, n), rng.lognormal(0.0, 0.6, n))
    mimic = fraud & (rng.random(n) < 0.35)
    mult = np.where(mimic, rng.lognormal(0.0, 0.6, n), mult)
    amount = np.round(np.clip(avg_amount * mult, 1, 50_000), 2)

    # --- time ----------------------------------------------------------------
    hour_w = np.array([1, .5, .4, .3, .3, .5, 1.5, 3, 4, 5, 5, 5.5,
                       6, 5.5, 5, 5, 5, 5.5, 6, 6, 5, 3.5, 2.5, 1.5])
    hours = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    night = fraud & (rng.random(n) < 0.45)
    hours[night] = rng.integers(0, 6, night.sum())
    ts = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(rng.integers(0, 60, n), unit="m")
    )

    # --- categorical attributes ---------------------------------------------
    merchant = _class_conditional_choice(
        rng, CATEGORICAL_LEVELS["merchant_category"],
        [.22, .18, .12, .05, .06, .10, .08, .07, .12],
        [.04, .05, .05, .18, .30, .08, .06, .02, .22], fraud)
    channel = _class_conditional_choice(
        rng, CATEGORICAL_LEVELS["payment_channel"],
        [.45, .30, .20, .05], [.10, .60, .22, .08], fraud)
    online_dev = rng.choice(["desktop", "mobile", "tablet"], size=n, p=[.5, .4, .1])
    app_dev = rng.choice(["mobile", "tablet"], size=n, p=[.9, .1])
    device = np.select(
        [np.isin(channel, ["pos", "atm"]), channel == "online"],
        ["pos_terminal", online_dev], default=app_dev)

    # --- behaviour / location signals ---------------------------------------
    is_new_device = (rng.random(n) < np.where(fraud, 0.45, 0.04)).astype(int)
    country_mismatch = (rng.random(n) < np.where(fraud, 0.28, 0.02)).astype(int)
    distance = np.where(fraud & (rng.random(n) < 0.5),
                        rng.exponential(350, n), rng.exponential(12, n)).round(1)
    base_age = rng.gamma(2.0, 450.0, n)
    account_age = np.where(fraud & (rng.random(n) < 0.35),
                           rng.integers(1, 45, n), base_age).astype(int).clip(min=1)
    txn_24h = rng.poisson(1.8, n) + np.where(fraud & (rng.random(n) < 0.4), rng.poisson(5, n), 0)
    since_last = np.where(fraud & (rng.random(n) < 0.4),
                          rng.exponential(6, n), rng.exponential(420, n)).round(1).clip(min=0.1)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(n)],
        "customer_id": [f"CUST{c:05d}" for c in customer],
        "timestamp": ts,
        "amount": amount,
        "merchant_category": merchant,
        "payment_channel": channel,
        "device_type": device,
        "is_new_device": is_new_device,
        "account_age_days": account_age,
        "distance_from_home_km": distance,
        "country_mismatch": country_mismatch,
        "txn_count_24h": txn_24h,
        "avg_amount_30d": avg_amount.round(2),
        "time_since_last_txn_min": since_last,
        TARGET: y,
    })

    # --- realistic data-quality issues so the cleaning steps have work to do ---
    for col in ("avg_amount_30d", "distance_from_home_km"):
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    n_dup = int(n * duplicate_rate)
    if n_dup:
        df = pd.concat([df, df.sample(n_dup, random_state=seed)], ignore_index=True)

    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate a synthetic fraud dataset.")
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_DATA)
    args = parser.parse_args()

    df = generate_transactions(args.rows, args.fraud_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %s rows (fraud rate %.2f%%) to %s",
                len(df), 100 * df[TARGET].mean(), args.output)


if __name__ == "__main__":
    main()
