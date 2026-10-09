"""Inference: probability -> 0-100 risk score -> risk level, plus SHAP explanation."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    RAW_REQUIRED_COLUMNS,
    RiskThresholds,
    get_decision_threshold_override,
    get_model_dir,
)
from src.explainability import FraudExplainer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Risk scoring (pure functions - easy to unit test)
# --------------------------------------------------------------------------- #
def probability_to_risk_score(probability: float) -> int:
    """Map a fraud probability in [0, 1] to an integer score in [0, 100]."""
    if probability is None or math.isnan(probability):
        raise ValueError("probability must be a number")
    p = min(max(float(probability), 0.0), 1.0)
    return int(math.floor(p * 100 + 0.5))  # round half up (avoids banker's rounding)


def assign_risk_level(risk_score: int, thresholds: RiskThresholds | None = None) -> str:
    """LOW / MEDIUM / HIGH using configurable bands (default 0-30 / 31-70 / 71-100)."""
    t = thresholds or RiskThresholds.from_env()
    if risk_score <= t.low_max:
        return "LOW"
    if risk_score <= t.medium_max:
        return "MEDIUM"
    return "HIGH"


# --------------------------------------------------------------------------- #
class ModelNotLoadedError(RuntimeError):
    pass


class FraudPredictor:
    """Loads saved artifacts once and scores transactions.

    The exact same preprocessing pipeline object that was fitted during training
    is reused here, which is what guarantees training/inference consistency.
    """

    def __init__(self, model: Any, preprocessor: Any, metadata: dict,
                 thresholds: RiskThresholds | None = None):
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.thresholds = thresholds or RiskThresholds.from_env()
        self.model_version: str = metadata["model_version"]
        self.decision_threshold: float = (
            get_decision_threshold_override() or float(metadata["decision_threshold"])
        )
        self.feature_names: list[str] = metadata["transformed_feature_names"]
        self.explainer = FraudExplainer(model, self.feature_names)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             thresholds: RiskThresholds | None = None) -> "FraudPredictor":
        model_dir = Path(model_dir) if model_dir else get_model_dir()
        paths = {
            "model": model_dir / MODEL_FILENAME,
            "preprocessor": model_dir / PREPROCESSOR_FILENAME,
            "metadata": model_dir / METADATA_FILENAME,
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise ModelNotLoadedError(
                f"Model artifacts not found: {missing}. Train first with `python -m src.train`."
            )
        metadata = json.loads(paths["metadata"].read_text())
        predictor = cls(joblib.load(paths["model"]), joblib.load(paths["preprocessor"]),
                        metadata, thresholds)
        logger.info("Loaded model %s from %s", predictor.model_version, model_dir)
        return predictor

    # ------------------------------------------------------------------ #
    def _score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        engineered = self.preprocessor.named_steps["features"].transform(df)
        X = self.preprocessor.named_steps["prep"].transform(engineered)
        proba = self.model.predict_proba(X)[:, 1]
        return proba, X, engineered

    def _build_result(self, probability: float) -> dict:
        score = probability_to_risk_score(probability)
        return {
            "fraud_probability": round(float(probability), 6),
            "risk_score": score,
            "risk_level": assign_risk_level(score, self.thresholds),
            "prediction": int(probability >= self.decision_threshold),
            "model_version": self.model_version,
        }

    def predict_dataframe(self, df: pd.DataFrame, explain: bool = False,
                          top_k: int = 5) -> list[dict]:
        proba, X, engineered = self._score(df)
        results = []
        for i, p in enumerate(proba):
            result = self._build_result(float(p))
            if explain:
                result.update(self.explainer.explain_row(
                    engineered.iloc[i], X[i], float(p), result["risk_level"], top_k))
            results.append(result)
        return results

    def predict(self, record: dict, explain: bool = True, top_k: int = 5) -> dict:
        return self.predict_dataframe(pd.DataFrame([record]), explain=explain, top_k=top_k)[0]

    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        m = self.metadata
        return {
            "model_version": self.model_version,
            "model_type": m["model_type"],
            "trained_at": m["trained_at"],
            "decision_threshold": self.decision_threshold,
            "risk_thresholds": self.thresholds.as_dict(),
            "test_metrics": m["test_metrics"],
            "comparison": m["comparison"],
            "global_importance": m["global_importance"],
            "class_distribution": m["class_distribution"],
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
