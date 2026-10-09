"""Training pipeline.

    python -m src.train                     # train on data/raw/transactions.csv
    python -m src.train --generate-if-missing
    python -m src.train --fast              # tiny search, for smoke tests

Leakage controls (the part interviewers ask about):
  1. Stratified train/test split happens FIRST, on raw rows.
  2. Imputer, scaler, encoder and SMOTENC live inside one imblearn Pipeline, so
     during cross-validation they are re-fitted on each training fold only and
     the validation fold is never resampled or used for fitting.
  3. The test set is touched exactly once, after model selection, for reporting.
  4. The decision threshold and model choice use out-of-fold (OOF) predictions
     on the TRAINING data, never the test set.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import evaluation as ev
from src.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_RAW_DATA,
    MODEL_FILENAME,
    MODEL_VERSION_BASE,
    METADATA_FILENAME,
    PREPROCESSOR_FILENAME,
    RANDOM_SEED,
    REPORTS_DIR,
    SMOTE_K_NEIGHBORS,
    SMOTE_SAMPLING_STRATEGY,
    TARGET,
    get_model_dir,
    setup_logging,
)
from src.explainability import FraudExplainer, plot_global_importance
from src.feature_engineering import FeatureEngineer
from src.preprocessing import (
    build_column_transformer,
    class_distribution,
    clean_dataset,
    load_dataset,
    n_output_features,
    split_data,
)

logger = logging.getLogger(__name__)

# Only tree models can be explained by TreeExplainer, so only they are eligible to
# be deployed. Logistic Regression stays in the comparison as an interpretable baseline.
DEPLOYABLE = {"xgboost", "random_forest"}


# --------------------------------------------------------------------------- #
# Models & search spaces
# --------------------------------------------------------------------------- #
def build_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed),
    }


def param_grids(fast: bool) -> dict:
    if fast:
        return {
            "logistic_regression": {"clf__C": [0.1, 1.0]},
            "random_forest": {"clf__n_estimators": [40], "clf__max_depth": [8, None]},
            "xgboost": {"clf__n_estimators": [60], "clf__max_depth": [3, 5],
                        "clf__learning_rate": [0.1]},
        }
    return {
        "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "random_forest": {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", 0.3],
        },
        "xgboost": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 0.5, 1.0],
        },
    }


def make_pipeline(clf, seed: int) -> ImbPipeline:
    """raw frame -> features -> impute/scale/encode -> SMOTENC -> classifier.

    SMOTENC (a SMOTE variant for mixed data) is used instead of plain SMOTE because
    most columns after encoding are binary/one-hot; plain SMOTE would interpolate
    them into meaningless fractions like 0.37 of "new device". Resampling steps
    are only applied by imblearn during `fit`, never during predict.
    """
    n_num, n_total = n_output_features()
    return ImbPipeline([
        ("features", FeatureEngineer()),
        ("prep", build_column_transformer()),
        ("smote", SMOTENC(categorical_features=list(range(n_num, n_total)),
                          sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                          k_neighbors=SMOTE_K_NEIGHBORS, random_state=seed)),
        ("clf", clf),
    ])


def _grid_size(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


# --------------------------------------------------------------------------- #
def run_training(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    reports_dir: Path = REPORTS_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    seed: int = RANDOM_SEED,
    fast: bool = False,
    n_iter: int = 12,
    cv_folds: int = 3,
    beta: float = 2.0,
    explain_sample: int = 2000,
) -> dict:
    output_dir = Path(output_dir) if output_dir else get_model_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1-5. clean, inspect, split --------------------------------------------
    df, cleaning_report = clean_dataset(df)
    dist = class_distribution(df)
    logger.info("Class distribution: %s", dist)
    X_train, X_test, y_train, y_test = split_data(df, seed=seed)
    pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / "test.csv", index=False)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grids = param_grids(fast)
    models = build_models(seed)

    fitted: dict[str, Pipeline] = {}
    best_params: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    rows: list[dict] = []
    selection: dict[str, float] = {}

    # 12-13. train + tune each model ----------------------------------------
    for name, clf in models.items():
        logger.info("=== %s ===", name)
        grid = grids[name]
        search = RandomizedSearchCV(
            make_pipeline(clf, seed), grid,
            n_iter=min(n_iter, _grid_size(grid)),
            scoring="average_precision",  # PR-AUC: the right target under heavy imbalance
            cv=cv, random_state=seed, n_jobs=1, refit=True,
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        best_params[name] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        logger.info("best CV PR-AUC=%.4f params=%s", search.best_score_, best_params[name])

        # Out-of-fold probabilities on TRAIN data -> threshold + model selection.
        oof[name] = cross_val_predict(
            clone(search.best_estimator_), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thresholds[name] = ev.optimal_threshold(y_train, oof[name], beta=beta)
        val_m = ev.compute_metrics(y_train, oof[name], thresholds[name])
        selection[name] = ev.selection_score(val_m)
        rows.append({"model": name, "split": "validation (OOF)", **val_m})

    # 14. held-out test evaluation (reporting only) --------------------------
    test_proba = {n: fitted[n].predict_proba(X_test)[:, 1] for n in fitted}
    test_metrics = {n: ev.compute_metrics(y_test, test_proba[n], thresholds[n]) for n in fitted}
    for n, m in test_metrics.items():
        rows.append({"model": n, "split": "test", **m})
    table = ev.comparison_table(rows)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    logger.info("\n%s", table.round(4).to_string(index=False))

    # 15. select the best deployable model -----------------------------------
    best_overall = max(selection, key=selection.get)
    best_name = max((n for n in selection if n in DEPLOYABLE), key=selection.get)
    logger.info("Best overall: %s | deployed (explainable): %s", best_overall, best_name)

    # 16. extract + save fitted artifacts ------------------------------------
    best_pipe = fitted[best_name]
    preprocessor = Pipeline([("features", best_pipe.named_steps["features"]),
                             ("prep", best_pipe.named_steps["prep"])])
    model = best_pipe.named_steps["clf"]
    feature_names = list(preprocessor.named_steps["prep"].get_feature_names_out())

    # 20. SHAP global importance on a test sample ----------------------------
    sample = X_test.sample(min(explain_sample, len(X_test)), random_state=seed)
    engineered_sample = preprocessor.named_steps["features"].transform(sample)
    X_sample = preprocessor.named_steps["prep"].transform(engineered_sample)
    importance = FraudExplainer(model, feature_names).global_importance(X_sample)
    plot_global_importance(importance, reports_dir / "shap_global_importance.png")

    ev.plot_confusion_matrices(test_metrics, reports_dir / "confusion_matrices.png")
    ev.plot_curves(y_test, test_proba, reports_dir / "pr_curves.png", reports_dir / "roc_curves.png")

    trained_at = datetime.now(timezone.utc)
    version = f"{MODEL_VERSION_BASE}-{trained_at:%Y%m%d%H%M%S}"
    metadata = {
        "model_version": version,
        "model_type": {"xgboost": "XGBoost", "random_forest": "Random Forest"}[best_name],
        "model_key": best_name,
        "trained_at": trained_at.isoformat(),
        "decision_threshold": thresholds[best_name],
        "threshold_beta": beta,
        "best_params": best_params[best_name],
        "best_overall_by_selection_score": best_overall,
        "selection_scores": selection,
        "test_metrics": test_metrics[best_name],
        "comparison": json.loads(table.to_json(orient="records")),
        "class_distribution": dist,
        "cleaning_report": cleaning_report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "smote": {"type": "SMOTENC", "sampling_strategy": SMOTE_SAMPLING_STRATEGY,
                  "k_neighbors": SMOTE_K_NEIGHBORS, "applied_to": "training folds only"},
        "transformed_feature_names": feature_names,
        "global_importance": importance,
        "seed": seed,
    }

    joblib.dump(model, output_dir / MODEL_FILENAME, compress=3)
    joblib.dump(preprocessor, output_dir / PREPROCESSOR_FILENAME, compress=3)
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, default=str))

    # keep an immutable, versioned copy for rollback / audit
    archive = output_dir / "versions" / version
    archive.mkdir(parents=True, exist_ok=True)
    for fname in (MODEL_FILENAME, PREPROCESSOR_FILENAME, METADATA_FILENAME):
        shutil.copy2(output_dir / fname, archive / fname)

    logger.info("Saved model %s to %s", version, output_dir)
    return metadata


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Train the fraud detection models.")
    p.add_argument("--data", type=Path, default=DEFAULT_RAW_DATA)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fast", action="store_true", help="tiny hyper-parameter search")
    p.add_argument("--n-iter", type=int, default=12)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--beta", type=float, default=2.0,
                   help="F-beta used to pick the decision threshold (>1 favours recall)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--generate-if-missing", action="store_true",
                   help="create the synthetic demo dataset if --data does not exist")
    args = p.parse_args()

    if args.generate_if_missing and not args.data.exists():
        from src.data_generation import generate_transactions

        logger.info("Dataset missing - generating synthetic data at %s", args.data)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        generate_transactions(seed=args.seed).to_csv(args.data, index=False)

    df = load_dataset(args.data)
    meta = run_training(df, output_dir=args.output_dir, seed=args.seed, fast=args.fast,
                        n_iter=args.n_iter, cv_folds=args.cv_folds, beta=args.beta)
    logger.info("Done. Deployed %s %s | test recall=%.3f precision=%.3f PR-AUC=%.3f",
                meta["model_type"], meta["model_version"], meta["test_metrics"]["recall"],
                meta["test_metrics"]["precision"], meta["test_metrics"]["pr_auc"])


if __name__ == "__main__":
    main()
