# 🛡️ Fraud Detection & Real-Time Risk Scoring System

**XGBoost · SHAP · FastAPI · PostgreSQL · Streamlit · Docker**

An end-to-end machine-learning system that scores financial transactions in real time, turns the fraud probability into a **0–100 risk score and LOW / MEDIUM / HIGH level**, and **explains every decision with SHAP**. Includes the full ML pipeline (EDA → SMOTE → model comparison → artifacts), a REST API, a database, an analytics dashboard, tests and Docker deployment.

> **Status note.** The synthetic dataset, pipeline, API and dashboard are complete. Model metrics are produced by *your* run of `python -m src.train` — the [Project Results](#project-results) section is intentionally a template to fill in with your own numbers.

---

## Table of contents
1. [Problem statement](#problem-statement) · 2. [Objectives](#objectives) · 3. [Features](#features) · 4. [Architecture](#architecture) · 5. [Tech stack](#tech-stack) · 6. [Dataset](#dataset) · 7. [Preprocessing](#data-preprocessing) · 8. [Feature engineering](#feature-engineering) · 9. [SMOTE](#smote) · 10. [Models](#ml-models) · 11. [Evaluation](#model-evaluation) · 12. [Risk scoring](#risk-scoring) · 13. [SHAP](#shap-explainability) · 14. [API](#api-documentation) · 15. [Dashboard](#dashboard) · 16. [Database](#database) · 17. [Docker](#docker-setup) · 18. [Installation](#installation) · 19. [Usage](#usage) · 20. [Deployment](#deployment) · 21. [Screenshots](#screenshots) · 22. [Future improvements](#future-improvements) · 23. [Results](#project-results)

---

## Problem statement
Card and online-payment fraud is rare (typically well under 1–2% of transactions) but expensive. Analysts cannot review everything, and a black-box "fraud / not fraud" label is not actionable: reviewers need to know **how risky** a transaction is and **why**. Rarity also breaks naive evaluation — a model that always says "legitimate" is ~98% accurate and catches nothing.

## Objectives
- Detect fraudulent transactions with high **recall** while keeping false alarms manageable.
- Produce a calibrated-for-triage **risk score** (0–100) and level with configurable bands.
- Explain each prediction from the model's actual SHAP values.
- Serve predictions through a validated API, persist them, and visualise them.
- Be reproducible, leakage-free, tested, and deployable with one command.

## Features
- Full ML pipeline: cleaning, EDA, feature engineering, stratified split, **SMOTENC on training data only**, 3 models with hyper-parameter search, PR-AUC-driven selection
- Out-of-fold threshold selection (F-beta, recall-weighted) — the test set is touched once
- Configurable risk bands via environment variables
- Per-prediction SHAP: top features increasing / decreasing risk + generated explanation text
- FastAPI: `/health`, `/predict`, `/batch-predict`, `/model-info` (+ `/history`, `/stats`), Pydantic validation, optional API-key auth
- PostgreSQL storage of every prediction (SQLAlchemy 2)
- Streamlit dashboard: overview, prediction form, analytics (Plotly), filterable history
- Model versioning (`models/versions/<version>/` + version stored with each prediction)
- Pytest suite, Docker Compose, Render blueprint

## Architecture
```
Transaction data (CSV)
        ↓
Data preprocessing        clean · dedupe · impute · scale · one-hot
        ↓
Feature engineering       stateless, row-wise derived features
        ↓
SMOTENC                   training folds only
        ↓
Model training            Logistic Regression · Random Forest · XGBoost (tuned)
        ↓
Model evaluation          Precision · Recall · F1 · ROC-AUC · PR-AUC · confusion matrix
        ↓
Best model (tree-based)   usually XGBoost
        ↓
SHAP explainability
        ↓
FastAPI  ──────►  PostgreSQL
        ↓
Streamlit dashboard (calls the API)
```

## Tech stack
| Area | Tools |
|---|---|
| Language | Python 3.11+, Jupyter, VS Code |
| Data / viz | Pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTENC), SHAP, Joblib |
| Backend | FastAPI, Pydantic, Uvicorn, SQLAlchemy, PostgreSQL |
| Frontend | Streamlit, Plotly |
| Ops | Docker, Docker Compose, Render / AWS, Pytest, Git |

## Dataset
**The bundled generator creates *synthetic* data — it is not real banking or customer data.**

Why synthetic? Popular public datasets (e.g. Kaggle *Credit Card Fraud Detection*) contain only anonymised PCA components (`V1…V28`), `Time` and `Amount`. They have no device, location or behaviour fields, so they cannot drive a meaningful transaction form or human-readable explanations. `src/data_generation.py` produces a table with the same *kinds* of signals analysts use, with deliberate class overlap, plus injected missing values and duplicates so cleaning is exercised.

| Column | Meaning |
|---|---|
| `transaction_id`, `customer_id` | identifiers (dropped before modelling) |
| `timestamp` | local time of the transaction |
| `amount` | transaction amount |
| `merchant_category`, `payment_channel`, `device_type` | categorical attributes |
| `is_new_device` | first time this device is seen for the customer |
| `account_age_days` | customer behaviour |
| `distance_from_home_km`, `country_mismatch` | location information |
| `txn_count_24h`, `time_since_last_txn_min`, `avg_amount_30d` | frequency / behaviour aggregates (supplied by the caller; in production from a feature store) |
| **`is_fraud`** | **target: 0 = legitimate, 1 = fraudulent** |

**Using a real dataset:** map its columns onto the schema above (see `RAW_REQUIRED_COLUMNS` in `src/config.py`), save it as `data/raw/transactions.csv`, and run `python -m src.train`. If your data lacks a field, derive it only from available columns and document it; do not fabricate values.

## Data preprocessing
1. Load CSV → schema validation (required columns, target ∈ {0,1}).
2. Check missing values (reported; median-imputed inside the pipeline, **fitted on training data only**).
3. Remove exact duplicates.
4. Analyse class distribution (fraud rate, imbalance ratio).
5. Detect outliers with Tukey fences (reported only). **Outliers are not removed** — in fraud, extreme amounts/velocity are the signal.
6. Encode categoricals with a one-hot encoder using fixed category lists (unseen levels → all zeros, never a crash).
7. Scale numerics with `StandardScaler` (needed by Logistic Regression, harmless for trees).
8. **Stratified** 80/20 split on raw rows *before* any resampling.

The identical fitted `Pipeline(FeatureEngineer → ColumnTransformer)` is saved to `models/preprocessing_pipeline.pkl` and reused at inference, so training and serving cannot diverge.

## Feature engineering
Only derived from columns that already exist; stateless and row-wise (no leakage):

| Feature | Definition |
|---|---|
| `amount_log`, `distance_log` | `log1p` of amount / distance |
| `amount_to_avg_ratio` | `amount / (avg_amount_30d + 1)` — spend relative to habit |
| `hour`, `is_night` | hour of day; 00:00–05:59 |
| `is_weekend` | Saturday / Sunday |
| `is_rapid_succession` | < 5 minutes since previous transaction |
| `is_high_velocity` | ≥ 6 transactions in 24 h |
| `is_new_account` | account age < 30 days |

Thresholds are constants in `src/config.py`.

## SMOTE
Class imbalance is handled with **SMOTENC** (SMOTE for mixed continuous/categorical features) because most encoded columns are binary/one-hot — plain SMOTE would interpolate them into meaningless fractions. Key decisions (commented in code):
- Applied **only to training data**, and *inside* the imblearn pipeline so each cross-validation fold resamples only its own training portion. The validation fold and test set are never resampled.
- Oversampling target is 20% of the majority class (`SMOTE_SAMPLING_STRATEGY`), not 1:1; full balancing inflates predicted probabilities and over-flags fraud.
- Because SMOTE shifts the probability scale, treat the risk score as a **ranking/triage score**, not a calibrated probability (see [Future improvements](#future-improvements)).

## ML models
| Model | Role | Tuned parameters |
|---|---|---|
| Logistic Regression | interpretable baseline | `C` |
| Random Forest | non-linear baseline | trees, depth, min leaf, max features |
| XGBoost | primary candidate | trees, depth, learning rate, subsample, colsample, min child weight, gamma |

`RandomizedSearchCV` (stratified 3-fold, scoring = **average precision / PR-AUC**) tunes each model. Only tree models are eligible for deployment, because the explainer uses `shap.TreeExplainer`; Logistic Regression remains in the comparison table.

## Model evaluation
Accuracy is never used. Every model is reported with Precision, Recall, F1, ROC-AUC, PR-AUC and a confusion matrix (`reports/`).

1. **Out-of-fold (OOF) predictions on the training set** give each model's decision threshold (maximising F-β, β=2 by default → recall-weighted because a missed fraud costs more than a manual review) and its **selection score = mean(Recall, F1, PR-AUC)**.
2. The model with the best selection score among deployable models is saved.
3. **Held-out test set** metrics are computed once at the end for reporting only.

Output: `reports/model_comparison.csv`, `confusion_matrices.png`, `pr_curves.png`, `roc_curves.png`, `shap_global_importance.png`.

## Risk scoring
```
Transaction → XGBoost → fraud_probability = 0.87 → risk_score = 87 → HIGH
```
| Score | Level |
|---|---|
| 0–30 | LOW |
| 31–70 | MEDIUM |
| 71–100 | HIGH |

Bands are configurable (`RISK_LOW_MAX`, `RISK_MEDIUM_MAX`) and validated at startup. The binary `prediction` (0/1) uses a separate decision threshold learned at training time (override with `DECISION_THRESHOLD`); the risk level is a graded triage band, so a transaction can be flagged `prediction=1` while still `MEDIUM`.

## SHAP explainability
Every `/predict` response includes the top features **increasing** and **decreasing** the fraud probability (SHAP log-odds contributions; one-hot columns are summed back to their source field) and a sentence generated from those values, for example:

> Fraud probability is 91% (HIGH risk). Amount vs. 30-day average (39.3x), New device (yes), Transaction hour (03:00) contributed most strongly to the elevated risk. Account age (900 days) reduced the risk.

*(Illustrative format; the text is composed at runtime from the real SHAP output, never hard-coded.)* Global importance (mean |SHAP|) is stored in the model metadata and shown in the dashboard.

## API documentation
Interactive docs at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: model loaded? database reachable? |
| POST | `/predict` | score one transaction (+ SHAP explanation), stores result |
| POST | `/batch-predict` | score up to 1000 transactions; `include_explanations` optional |
| GET | `/model-info` | version, metrics, comparison, thresholds, global importance |
| GET | `/history` | stored predictions with filters (date, risk level, status, amount) |
| GET | `/stats` | KPIs, risk distribution, daily trend |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "amount": 1250.0, "merchant_category": "electronics", "payment_channel": "online",
  "device_type": "desktop", "is_new_device": true, "account_age_days": 12,
  "distance_from_home_km": 820.5, "country_mismatch": true, "txn_count_24h": 7,
  "avg_amount_30d": 85.0, "time_since_last_txn_min": 2.5, "timestamp": "2026-03-14T03:20:00"
}'
```
Response fields: `transaction_id, fraud_probability, risk_score, risk_level, prediction, prediction_label, model_version, top_increasing_features[], top_decreasing_features[], explanation, stored`.

Validation: amounts must be > 0, categories must be known values, unknown fields are rejected (422). If the DB write fails the prediction is still returned with `stored: false`. Set `API_KEY` to require an `X-API-Key` header on everything except `/health`.

## Dashboard
`http://localhost:8501`
- **Overview** — total, fraud count, fraud %, average risk score, high-risk count, risk mix, daily volume
- **Transaction Prediction** — form (with *typical / suspicious* presets), prediction, probability, risk gauge, SHAP bar chart and explanation
- **Analytics** — fraud vs legitimate, risk distribution, amount distribution, fraud trend, model performance, confusion matrix, top fraud-related features
- **Transaction History** — filters for date, risk level, fraud status and amount; CSV download

## Database
Table `predictions`: `transaction_id, timestamp` (scored at), `transaction_time`, `amount`, `input_data` (JSON/JSONB — full input), `fraud_probability, risk_score, risk_level, prediction, model_version`. Configured with `POSTGRES_*` or `DATABASE_URL`; with neither set the app uses a local SQLite file for development/tests only.

## Docker setup
```bash
cp .env.example .env            # set POSTGRES_PASSWORD (and optionally API_KEY)
docker compose --profile train run --rm trainer   # generates demo data + trains -> ./models
docker compose up --build       # db + api + dashboard
```
API → <http://localhost:8000/docs> · Dashboard → <http://localhost:8501>

Services: `db` (postgres:16), `api` (FastAPI), `dashboard` (Streamlit), and an opt-in `trainer` job. `./models` is bind-mounted into the API so retraining needs no rebuild (restart `api` to load the new model).

## Installation
```bash
git clone <your-repo-url> fraud-detection-system && cd fraud-detection-system
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage
```bash
python -m src.data_generation          # or place your own CSV at data/raw/transactions.csv
python -m src.train                    # add --fast for a quick smoke run
pytest                                  # run the tests

# local API (SQLite fallback) and dashboard
uvicorn api.main:app --reload
streamlit run dashboard/app.py
python scripts/seed_demo.py --n 1500    # populate the dashboard with scored demo transactions
```
Notebooks (`jupyter lab`): `01_eda` → `02_preprocessing` → `03_model_training`. They ship **without outputs**; run them to generate yours. Shortcuts for these commands are in the `Makefile`.

## Deployment
**Render (Docker).** Push to GitHub, commit `models/*.pkl` and `models/model_metadata.json` (they are small and the image bakes them in), then *New → Blueprint* and select the repo (`render.yaml` provisions Postgres, the API and the dashboard). `render.yaml` has not been deployed by the author — verify against Render's blueprint spec if it is rejected.

**AWS (outline).** Build and push both images to ECR → run them as two ECS Fargate services behind an Application Load Balancer → use RDS PostgreSQL → inject `POSTGRES_*`/`API_KEY` from Secrets Manager → set the dashboard's `API_URL` to the API's internal service URL → restrict the API to the dashboard's security group (or keep `API_KEY` on).

## Screenshots
| | |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Prediction](docs/screenshots/prediction.png) |
| ![Analytics](docs/screenshots/analytics.png) | ![History](docs/screenshots/history.png) |

*(Placeholders — capture these after running the app and save them under `docs/screenshots/`.)*

## Future improvements
- **Probability calibration** (isotonic/Platt on an untouched calibration set) so the 0–100 score is a true probability despite SMOTE.
- Real streaming features (rolling per-customer aggregates from Redis/Kafka) instead of caller-supplied aggregates.
- Cost-sensitive thresholds from actual fraud-loss and review-cost figures.
- Drift monitoring (PSI) and scheduled retraining; champion/challenger rollout.
- Analyst feedback loop (confirmed fraud labels written back to the DB).
- Authentication/rate limiting beyond the shared API key; Alembic migrations.
- Evaluation on a real public dataset and on time-based splits.

## Project results
Fill this in from your own run (`reports/model_comparison.csv`, `models/model_metadata.json`):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | _run_ | _run_ | _run_ | _run_ | _run_ |
| Random Forest | _run_ | _run_ | _run_ | _run_ | _run_ |
| XGBoost | _run_ | _run_ | _run_ | _run_ | _run_ |

Remember to state in interviews that these results are on **synthetic** data, which is considerably easier than real-world fraud (even a plain Random Forest reaches ROC-AUC ≈ 0.99 on it), so absolute numbers say little about production performance — what matters is the leakage-free methodology. To see realistic difficulty, run the pipeline on a real public dataset.

---

### Interview talking points
- *Why not accuracy?* Imbalance; always-legit ≈ 98% accurate. Use PR-AUC, recall, F1.
- *Why SMOTE after the split / inside CV?* Otherwise synthetic copies of test-like rows leak into training and inflate scores.
- *Why SMOTENC?* One-hot/binary features can't be interpolated.
- *Why OOF predictions for the threshold?* Choosing it on the test set would make the test score optimistic.
- *Why not remove outliers?* In fraud they are the signal.
- *Why is the risk score not a probability?* SMOTE distorts calibration; fix with calibration (future work).
