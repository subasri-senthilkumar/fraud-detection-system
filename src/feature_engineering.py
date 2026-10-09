"""Feature engineering as a scikit-learn transformer.

Design rules
------------
* Stateless and row-wise: every derived feature depends only on columns of the
  *same row*. Nothing is learned from the data, so there is no train/test leakage
  and training/inference behave identically.
* Only columns that exist in the input are used. No customer history is invented:
  ``avg_amount_30d``, ``txn_count_24h`` etc. must be supplied by the caller
  (in production they would come from a feature store / aggregation job).

Derived features (all documented in the README):

=========================  ====================================================
amount_log                 log1p(amount) - tames the heavy right tail
distance_log               log1p(distance_from_home_km)
amount_to_avg_ratio        amount / (avg_amount_30d + 1) - spend vs. habit
hour                       hour of day from timestamp
is_night                   hour in 00:00-05:59
is_weekend                 Saturday/Sunday
is_rapid_succession        < RAPID_SUCCESSION_MINUTES since previous txn
is_high_velocity           >= HIGH_VELOCITY_TXN_24H transactions in 24h
is_new_account             account age < NEW_ACCOUNT_DAYS
=========================  ====================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    HIGH_VELOCITY_TXN_24H,
    NEW_ACCOUNT_DAYS,
    NIGHT_HOURS,
    NUMERIC_FEATURES,
    RAPID_SUCCESSION_MINUTES,
    RAW_REQUIRED_COLUMNS,
)

MODEL_INPUT_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn raw transaction columns into the model's engineered feature frame.

    Missing numeric values are left as NaN on purpose; the median imputer inside
    the ColumnTransformer fills them (fitted on training data only).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        self._check_columns(X)
        return self

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        df = X.copy()

        num = lambda col: pd.to_numeric(df[col], errors="coerce")  # noqa: E731

        amount = num("amount").clip(lower=0)
        avg_amount = num("avg_amount_30d")
        distance = num("distance_from_home_km").clip(lower=0)
        account_age = num("account_age_days")
        txn_count = num("txn_count_24h")
        since_last = num("time_since_last_txn_min")

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            # Keep the wall-clock time; "night" is about local time of the purchase.
            ts = ts.dt.tz_localize(None)
        hour = ts.dt.hour.astype("float64")  # NaN if timestamp unparsable

        out = pd.DataFrame(index=df.index)
        out["amount"] = amount
        out["amount_log"] = np.log1p(amount)
        out["account_age_days"] = account_age
        out["distance_from_home_km"] = distance
        out["distance_log"] = np.log1p(distance)
        out["txn_count_24h"] = txn_count
        out["avg_amount_30d"] = avg_amount
        out["amount_to_avg_ratio"] = amount / (avg_amount + 1.0)
        out["time_since_last_txn_min"] = since_last
        out["hour"] = hour

        out["is_new_device"] = num("is_new_device").fillna(0).clip(0, 1).astype(int)
        out["country_mismatch"] = num("country_mismatch").fillna(0).clip(0, 1).astype(int)
        out["is_night"] = hour.isin(list(NIGHT_HOURS)).astype(int)
        out["is_weekend"] = (ts.dt.dayofweek >= 5).fillna(False).astype(int)
        out["is_rapid_succession"] = (since_last < RAPID_SUCCESSION_MINUTES).fillna(False).astype(int)
        out["is_high_velocity"] = (txn_count >= HIGH_VELOCITY_TXN_24H).fillna(False).astype(int)
        out["is_new_account"] = (account_age < NEW_ACCOUNT_DAYS).fillna(False).astype(int)

        for col in CATEGORICAL_FEATURES:
            out[col] = df[col].astype("object")

        return out[MODEL_INPUT_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_INPUT_COLUMNS, dtype=object)
