"""Metrics, threshold selection, model-selection score and evaluation plots.

Accuracy is intentionally absent: with ~2% fraud, predicting "legitimate" for
everything scores ~98% accuracy while catching zero fraud.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def optimal_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Decision threshold maximising F-beta on the given (validation) data.

    Business trade-off: a missed fraud (false negative) usually costs far more
    than a needless review (false positive), so beta > 1 favours recall.
    beta=1 gives the balanced F1-optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    denom = beta**2 * precision + recall
    fbeta = np.divide((1 + beta**2) * precision * recall, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(fbeta))
    return float(np.clip(thresholds[best], 0.01, 0.99))


def selection_score(metrics: dict) -> float:
    """Composite used to pick the deployed model: mean of Recall, F1 and PR-AUC
    (the three criteria prioritised for fraud detection)."""
    return float(np.mean([metrics["recall"], metrics["f1"], metrics["pr_auc"]]))


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with keys model, split + metric keys."""
    df = pd.DataFrame(rows)
    out = df.rename(columns={
        "model": "Model", "split": "Split", "threshold": "Threshold",
        "precision": "Precision", "recall": "Recall", "f1": "F1",
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN",
    })
    cols = ["Model", "Split", "Threshold", "Precision", "Recall", "F1",
            "ROC-AUC", "PR-AUC", "TP", "FP", "FN", "TN"]
    return out[[c for c in cols if c in out.columns]]


# --------------------------------------------------------------------------- #
# Plots (saved as PNG so the README / reports can embed them)
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(results: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(y_true, probas: dict[str, np.ndarray], path_pr: Path, path_roc: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1, label="Baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test set)")
    ax.legend()
    fig.tight_layout()
    path_pr.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pr, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_roc, dpi=140)
    plt.close(fig)
