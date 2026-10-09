"""SHAP-based explanations.

All explanation text is *generated from the SHAP values of the actual
prediction* - nothing is hard-coded per outcome.

Units: for XGBoost, SHAP values are in log-odds (margin) space; positive values
push the fraud probability up, negative values push it down. One-hot columns are
summed back to their source feature (e.g. all `merchant_category_*` columns ->
`merchant_category`) so the explanation talks about the fields a human entered.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CATEGORICAL_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

# (human label, formatter) per engineered feature.
FEATURE_META: dict[str, tuple[str, str]] = {
    "amount": ("Transaction amount", "money"),
    "amount_log": ("Transaction amount (log)", "num"),
    "account_age_days": ("Account age", "days"),
    "distance_from_home_km": ("Distance from home", "km"),
    "distance_log": ("Distance from home (log)", "num"),
    "txn_count_24h": ("Transactions in last 24h", "int"),
    "avg_amount_30d": ("Customer 30-day average amount", "money"),
    "amount_to_avg_ratio": ("Amount vs. 30-day average", "ratio"),
    "time_since_last_txn_min": ("Minutes since previous transaction", "min"),
    "hour": ("Transaction hour", "hour"),
    "is_new_device": ("New device", "bool"),
    "country_mismatch": ("Country mismatch", "bool"),
    "is_night": ("Night-time transaction", "bool"),
    "is_weekend": ("Weekend transaction", "bool"),
    "is_rapid_succession": ("Rapid succession of transactions", "bool"),
    "is_high_velocity": ("High transaction frequency", "bool"),
    "is_new_account": ("New account", "bool"),
    "merchant_category": ("Merchant category", "text"),
    "payment_channel": ("Payment channel", "text"),
    "device_type": ("Device type", "text"),
}


def source_feature(transformed_name: str) -> str:
    """Map a model-matrix column (possibly one-hot) back to its source feature."""
    for cat in CATEGORICAL_FEATURES:
        if transformed_name.startswith(f"{cat}_"):
            return cat
    return transformed_name


def feature_label(name: str) -> str:
    return FEATURE_META.get(name, (name.replace("_", " ").capitalize(), "num"))[0]


def format_value(name: str, value: Any) -> str:
    kind = FEATURE_META.get(name, ("", "num"))[1]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    try:
        if kind == "bool":
            return "yes" if float(value) >= 0.5 else "no"
        if kind == "money":
            return f"{float(value):,.2f}"
        if kind == "ratio":
            return f"{float(value):.1f}x"
        if kind == "days":
            return f"{int(float(value)):,} days"
        if kind == "km":
            return f"{float(value):,.1f} km"
        if kind == "min":
            return f"{float(value):,.1f} min"
        if kind == "hour":
            return f"{int(float(value)):02d}:00"
        if kind == "int":
            return f"{int(float(value))}"
        if kind == "text":
            return str(value).replace("_", " ")
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


class FraudExplainer:
    """Wraps shap.TreeExplainer for tree models (XGBoost / Random Forest)."""

    def __init__(self, model: Any, transformed_feature_names: list[str]):
        self.model = model
        self.names = list(transformed_feature_names)
        self.sources = [source_feature(n) for n in self.names]
        self._explainer = None
        try:
            import shap

            self._explainer = shap.TreeExplainer(model)
        except Exception as exc:  # shap/xgboost version mismatches happen in practice
            logger.warning("shap.TreeExplainer unavailable (%s); will use native XGBoost "
                           "contributions, which are exact tree-SHAP values too.", exc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_fraud_class(values: Any) -> np.ndarray:
        if isinstance(values, list):  # older shap + sklearn forests: [class0, class1]
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # newer shap: (n, features, classes)
            values = values[:, :, 1]
        return values

    def _native_contribs(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        booster = self.model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        return contribs[:, :-1]  # last column is the bias term

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._explainer is not None:
            try:
                return self._to_fraud_class(self._explainer.shap_values(X))
            except Exception as exc:
                logger.warning("TreeExplainer failed (%s); falling back to native contribs", exc)
        return self._native_contribs(X)

    # ------------------------------------------------------------------ #
    def _aggregate(self, shap_row: np.ndarray) -> dict[str, float]:
        agg: dict[str, float] = {}
        for src, val in zip(self.sources, shap_row):
            agg[src] = agg.get(src, 0.0) + float(val)
        return agg

    def explain_row(
        self,
        engineered_row: pd.Series,
        x_row: np.ndarray,
        probability: float,
        risk_level: str,
        top_k: int = 5,
    ) -> dict:
        shap_row = self.shap_values(np.asarray(x_row, dtype=float).reshape(1, -1))[0]
        agg = self._aggregate(shap_row)

        contribs = [
            {
                "feature": feat,
                "label": feature_label(feat),
                "value": format_value(feat, engineered_row.get(feat)),
                "shap_value": round(val, 4),
            }
            for feat, val in agg.items()
        ]
        increasing = sorted((c for c in contribs if c["shap_value"] > 0),
                            key=lambda c: -c["shap_value"])[:top_k]
        decreasing = sorted((c for c in contribs if c["shap_value"] < 0),
                            key=lambda c: c["shap_value"])[:top_k]
        return {
            "top_increasing": increasing,
            "top_decreasing": decreasing,
            "explanation": build_explanation(probability, risk_level, increasing, decreasing),
        }

    def global_importance(self, X: np.ndarray, top_k: int | None = None) -> list[dict]:
        """Mean |SHAP| per source feature over a sample of rows."""
        values = self.shap_values(X)
        per_source: dict[str, float] = {}
        abs_vals = np.abs(values).mean(axis=0)
        for src, v in zip(self.sources, abs_vals):
            per_source[src] = per_source.get(src, 0.0) + float(v)
        ranked = sorted(per_source.items(), key=lambda kv: -kv[1])
        if top_k:
            ranked = ranked[:top_k]
        return [{"feature": f, "label": feature_label(f), "mean_abs_shap": round(v, 5)}
                for f, v in ranked]


def build_explanation(probability: float, risk_level: str,
                      increasing: list[dict], decreasing: list[dict]) -> str:
    """Compose a sentence from the actual top SHAP drivers of this prediction."""
    parts = [f"Fraud probability is {probability:.0%} ({risk_level} risk)."]

    def describe(items: list[dict], n: int = 3) -> str:
        return ", ".join(f"{c['label']} ({c['value']})" for c in items[:n])

    if increasing:
        verb = "contributed most strongly to the elevated risk" if risk_level != "LOW" \
            else "were the main upward pressures on the score"
        parts.append(f"{describe(increasing)} {verb}.")
    if decreasing:
        parts.append(f"{describe(decreasing)} reduced the risk.")
    if not increasing and not decreasing:
        parts.append("No feature had a material influence on this prediction.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_global_importance(importance: list[dict], path: Path, top_k: int = 15) -> None:
    items = importance[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(items) + 1.2))
    ax.barh([i["label"] for i in items], [i["mean_abs_shap"] for i in items], color="#2a6f97")
    ax.set(xlabel="Mean |SHAP value| (impact on log-odds of fraud)",
           title="Global feature importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_local_explanation(result: dict, path: Path | None = None):
    """Bar chart of the top increasing/decreasing features for one prediction."""
    items = result["top_increasing"] + result["top_decreasing"]
    items = sorted(items, key=lambda c: c["shap_value"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(items) + 1.2))
    ax.barh([f"{c['label']} = {c['value']}" for c in items],
            [c["shap_value"] for c in items],
            color=["#c1121f" if c["shap_value"] > 0 else "#2a6f97" for c in items])
    ax.axvline(0, c="black", lw=0.8)
    ax.set(xlabel="SHAP value (log-odds)", title="Why this prediction?")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig
