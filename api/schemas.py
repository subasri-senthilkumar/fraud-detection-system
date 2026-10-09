"""Pydantic request / response models (validation lives here)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Keep in sync with src.config.CATEGORICAL_LEVELS (a unit test enforces this).
MerchantCategory = Literal[
    "grocery", "restaurant", "fuel", "travel", "electronics",
    "clothing", "entertainment", "health", "online_services",
]
PaymentChannel = Literal["pos", "online", "mobile_app", "atm"]
DeviceType = Literal["mobile", "desktop", "tablet", "pos_terminal"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class TransactionIn(BaseModel):
    """A single transaction to score."""

    transaction_id: str | None = Field(None, max_length=64, description="Generated if omitted")
    customer_id: str | None = Field(None, max_length=64)
    timestamp: datetime | None = Field(None, description="Local time of purchase; defaults to now (UTC)")
    amount: float = Field(..., gt=0, le=1_000_000)
    merchant_category: MerchantCategory
    payment_channel: PaymentChannel
    device_type: DeviceType
    is_new_device: bool = Field(..., description="First time this device is seen for the customer")
    account_age_days: int = Field(..., ge=0, le=50_000)
    distance_from_home_km: float | None = Field(None, ge=0, le=25_000)
    country_mismatch: bool = Field(..., description="Transaction country differs from home country")
    txn_count_24h: int = Field(..., ge=0, le=10_000)
    avg_amount_30d: float | None = Field(None, ge=0, le=1_000_000)
    time_since_last_txn_min: float = Field(..., ge=0, le=1_000_000)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "amount": 1250.0, "merchant_category": "electronics", "payment_channel": "online",
            "device_type": "desktop", "is_new_device": True, "account_age_days": 12,
            "distance_from_home_km": 820.5, "country_mismatch": True, "txn_count_24h": 7,
            "avg_amount_30d": 85.0, "time_since_last_txn_min": 2.5,
            "timestamp": "2026-03-14T03:20:00"}},
    )


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: str
    shap_value: float = Field(..., description="Log-odds contribution (+ raises, - lowers fraud risk)")


class PredictionOut(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    prediction: Literal[0, 1] = Field(..., description="0 = legitimate, 1 = fraud")
    prediction_label: Literal["LEGITIMATE", "FRAUD"]
    model_version: str
    top_increasing_features: list[FeatureContribution] = []
    top_decreasing_features: list[FeatureContribution] = []
    explanation: str | None = None
    stored: bool = Field(True, description="False if the DB write failed (prediction still returned)")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionIn] = Field(..., min_length=1, max_length=1000)
    include_explanations: bool = False


class BatchOut(BaseModel):
    count: int
    fraud_count: int
    results: list[PredictionOut]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database_connected: bool
    model_version: str | None = None


class ModelInfoOut(BaseModel):
    model_version: str
    model_type: str
    trained_at: str
    decision_threshold: float
    risk_thresholds: dict[str, int]
    test_metrics: dict
    comparison: list[dict]
    global_importance: list[dict]
    class_distribution: dict
    n_features: int
    feature_names: list[str]


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: str
    timestamp: datetime
    transaction_time: datetime | None
    amount: float
    fraud_probability: float
    risk_score: int
    risk_level: RiskLevel
    prediction: int
    model_version: str
    input_data: dict


class HistoryOut(BaseModel):
    total: int
    items: list[HistoryItem]


class DailyTrend(BaseModel):
    day: date
    total: int
    fraud: int


class StatsOut(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_percentage: float
    average_risk_score: float
    high_risk_transactions: int
    risk_distribution: dict[str, int]
    daily_trend: list[DailyTrend]
