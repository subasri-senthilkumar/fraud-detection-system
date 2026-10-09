"""Unit tests for risk scoring, features, data handling and the predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_LEVELS, TARGET, RiskThresholds
from src.data_generation import generate_transactions
from src.feature_engineering import MODEL_INPUT_COLUMNS, FeatureEngineer
from src.preprocessing import build_preprocessing_pipeline, clean_dataset, split_data
from src.predict import assign_risk_level, probability_to_risk_score
from tests.conftest import BAD_TXN, GOOD_TXN


# ------------------------------------------------------------------ risk scoring
@pytest.mark.parametrize("prob,score", [(0.0, 0), (0.87, 87), (0.005, 1), (0.304, 30), (1.0, 100)])
def test_probability_to_risk_score(prob, score):
    assert probability_to_risk_score(prob) == score


def test_probability_is_clipped_and_validated():
    assert probability_to_risk_score(1.7) == 100
    assert probability_to_risk_score(-0.2) == 0
    with pytest.raises(ValueError):
        probability_to_risk_score(float("nan"))


@pytest.mark.parametrize("score,level", [
    (0, "LOW"), (30, "LOW"), (31, "MEDIUM"), (70, "MEDIUM"), (71, "HIGH"), (100, "HIGH")])
def test_default_risk_bands(score, level):
    assert assign_risk_level(score, RiskThresholds()) == level


def test_risk_thresholds_are_configurable(monkeypatch):
    assert assign_risk_level(50, RiskThresholds(low_max=10, medium_max=40)) == "HIGH"
    monkeypatch.setenv("RISK_LOW_MAX", "50")
    monkeypatch.setenv("RISK_MEDIUM_MAX", "90")
    assert assign_risk_level(45) == "LOW"  # falls back to env-configured thresholds
    assert assign_risk_level(95) == "HIGH"


@pytest.mark.parametrize("low,med", [(70, 30), (30, 30), (-1, 50), (30, 100)])
def test_invalid_thresholds_rejected(low, med):
    with pytest.raises(ValueError):
        RiskThresholds(low_max=low, medium_max=med)


# --------------------------------------------------------------- data + features
def test_generator_is_reproducible_and_imbalanced():
    a = generate_transactions(3000, 0.03, seed=1)
    b = generate_transactions(3000, 0.03, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert 0.01 < a[TARGET].mean() < 0.06
    assert a.duplicated().sum() > 0          # duplicates exist so cleaning has work to do
    assert a["avg_amount_30d"].isna().any()  # and so do missing values


def test_clean_dataset_removes_duplicates_only():
    df = generate_transactions(3000, 0.03, seed=2)
    cleaned, report = clean_dataset(df)
    assert report["duplicates_removed"] == df.duplicated().sum()
    assert not cleaned.duplicated().any()
    assert len(cleaned) == len(df) - report["duplicates_removed"]


def test_clean_dataset_rejects_bad_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(pd.DataFrame({"amount": [1]}))


def test_split_is_stratified():
    df, _ = clean_dataset(generate_transactions(5000, 0.04, seed=3))
    X_tr, X_te, y_tr, y_te = split_data(df, test_size=0.2, seed=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.005
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_feature_engineering_values():
    row = {**BAD_TXN, "is_new_device": 1, "country_mismatch": 1}
    out = FeatureEngineer().transform(pd.DataFrame([row]))
    r = out.iloc[0]
    assert list(out.columns) == MODEL_INPUT_COLUMNS
    assert r["hour"] == 3 and r["is_night"] == 1
    assert r["is_high_velocity"] == 1 and r["is_rapid_succession"] == 1 and r["is_new_account"] == 1
    assert r["amount_to_avg_ratio"] == pytest.approx(2400 / 61)
    day = FeatureEngineer().transform(pd.DataFrame([GOOD_TXN])).iloc[0]
    assert day["is_night"] == 0 and day["is_high_velocity"] == 0


def test_feature_engineering_keeps_missing_numerics_as_nan():
    row = {**GOOD_TXN, "avg_amount_30d": None, "distance_from_home_km": None}
    out = FeatureEngineer().transform(pd.DataFrame([row]))
    assert np.isnan(out.loc[0, "amount_to_avg_ratio"]) and np.isnan(out.loc[0, "distance_log"])


def test_feature_engineering_requires_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureEngineer().transform(pd.DataFrame([{"amount": 5}]))


def test_preprocessing_has_no_nans_and_handles_unseen_category():
    df, _ = clean_dataset(generate_transactions(3000, 0.04, seed=4))
    X_tr, X_te, _, _ = split_data(df)
    pipe = build_preprocessing_pipeline().fit(X_tr)
    weird = X_te.head(3).copy()
    weird["merchant_category"] = "never_seen_before"
    Xt = pipe.transform(weird)
    assert not np.isnan(Xt).any()
    assert Xt.shape[1] == len(pipe.named_steps["prep"].get_feature_names_out())


def test_api_literals_match_config():
    from typing import get_args
    from api import schemas

    assert list(get_args(schemas.MerchantCategory)) == CATEGORICAL_LEVELS["merchant_category"]
    assert list(get_args(schemas.PaymentChannel)) == CATEGORICAL_LEVELS["payment_channel"]
    assert list(get_args(schemas.DeviceType)) == CATEGORICAL_LEVELS["device_type"]


# --------------------------------------------------------- trained model (needs ML deps)
def test_artifacts_saved_with_version(trained_dir):
    for name in ("fraud_model.pkl", "preprocessing_pipeline.pkl", "model_metadata.json"):
        assert (trained_dir / name).exists()
    versions = list((trained_dir / "versions").iterdir())
    assert len(versions) == 1


def test_smote_not_applied_to_test_data(trained_dir, small_df):
    import json

    meta = json.loads((trained_dir / "model_metadata.json").read_text())
    cleaned, _ = clean_dataset(small_df)
    assert meta["n_test"] == pytest.approx(0.2 * len(cleaned), abs=2)
    test_csv = trained_dir.parent / "processed" / "test.csv"
    test_fraud_rate = pd.read_csv(test_csv)[TARGET].mean()
    assert test_fraud_rate == pytest.approx(cleaned[TARGET].mean(), abs=0.005)  # still imbalanced


def test_predict_output_contract(predictor):
    res = predictor.predict(GOOD_TXN)
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["risk_score"] == probability_to_risk_score(res["fraud_probability"])
    assert res["risk_level"] == assign_risk_level(res["risk_score"], predictor.thresholds)
    assert res["prediction"] in (0, 1)
    assert res["model_version"] == predictor.model_version


def test_suspicious_scores_higher_than_normal(predictor):
    assert predictor.predict(BAD_TXN)["fraud_probability"] > predictor.predict(GOOD_TXN)["fraud_probability"]


def test_explanation_is_derived_from_shap(predictor):
    res = predictor.predict(BAD_TXN, top_k=3)
    inc, dec = res["top_increasing"], res["top_decreasing"]
    assert inc and all(c["shap_value"] > 0 for c in inc)
    assert all(c["shap_value"] < 0 for c in dec)
    assert [c["shap_value"] for c in inc] == sorted((c["shap_value"] for c in inc), reverse=True)
    assert inc[0]["label"] in res["explanation"]          # text is built from the top driver
    assert f"{res['fraud_probability']:.0%}" in res["explanation"]


def test_batch_matches_single(predictor):
    batch = predictor.predict_dataframe(pd.DataFrame([GOOD_TXN, BAD_TXN]))
    for rec, b in zip((GOOD_TXN, BAD_TXN), batch):
        assert b["fraud_probability"] == pytest.approx(predictor.predict(rec, explain=False)["fraud_probability"])


def test_missing_field_raises(predictor):
    bad = {k: v for k, v in GOOD_TXN.items() if k != "amount"}
    with pytest.raises(ValueError):
        predictor.predict(bad)
