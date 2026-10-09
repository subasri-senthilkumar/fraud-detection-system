"""FastAPI service.

    uvicorn api.main:app --reload

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api import database as dbm
from api.schemas import (
    BatchOut, BatchRequest, HealthOut, HistoryOut, ModelInfoOut,
    PredictionOut, StatsOut, TransactionIn,
)
from src.config import setup_logging
from src.predict import FraudPredictor, ModelNotLoadedError

setup_logging()
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    try:
        dbm.init_engine()
    except Exception:
        logger.exception("Could not initialise the database; predictions will not be stored")
    try:
        app.state.predictor = FraudPredictor.load()
    except ModelNotLoadedError as exc:
        logger.error("%s", exc)
    except Exception:
        logger.exception("Unexpected error while loading the model")
    yield


app = FastAPI(
    title="Fraud Detection & Real-Time Risk Scoring API",
    version="1.0.0",
    description="Scores transactions with XGBoost and explains each decision with SHAP.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Dependencies / error handling
# --------------------------------------------------------------------------- #
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional shared-secret auth: enforced only when API_KEY is set in the env."""
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header")


def get_predictor(request: Request) -> FraudPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Model not loaded. Train a model with `python -m src.train`.")
    return predictor


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_model_record(txn: TransactionIn) -> tuple[str, dict]:
    """Pydantic payload -> (transaction_id, flat dict in the model's raw schema)."""
    transaction_id = txn.transaction_id or f"txn_{uuid.uuid4().hex[:12]}"
    ts = txn.timestamp or datetime.now(timezone.utc)
    ts = ts.replace(tzinfo=None)  # keep wall-clock time (see FeatureEngineer)
    record = txn.model_dump(exclude={"transaction_id", "customer_id"})
    record.update(
        timestamp=ts,
        is_new_device=int(txn.is_new_device),
        country_mismatch=int(txn.country_mismatch),
    )
    return transaction_id, record


def _to_db_row(txn: TransactionIn, transaction_id: str, record: dict, result: dict) -> dbm.PredictionRecord:
    return dbm.PredictionRecord(
        transaction_id=transaction_id,
        transaction_time=record["timestamp"],
        amount=txn.amount,
        input_data=txn.model_copy(update={"transaction_id": transaction_id}).model_dump(mode="json"),
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        prediction=result["prediction"],
        model_version=result["model_version"],
    )


def _to_response(transaction_id: str, result: dict, stored: bool) -> PredictionOut:
    return PredictionOut(
        transaction_id=transaction_id,
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        prediction=result["prediction"],
        prediction_label="FRAUD" if result["prediction"] == 1 else "LEGITIMATE",
        model_version=result["model_version"],
        top_increasing_features=result.get("top_increasing", []),
        top_decreasing_features=result.get("top_decreasing", []),
        explanation=result.get("explanation"),
        stored=stored,
    )


def _try_store(db: Session, rows: list[dbm.PredictionRecord]) -> bool:
    try:
        dbm.save_predictions(db, rows)
        return True
    except Exception:
        db.rollback()
        logger.exception("Failed to store %s prediction(s)", len(rows))
        return False


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut, tags=["system"])
def health(request: Request) -> HealthOut:
    predictor = getattr(request.app.state, "predictor", None)
    db_ok = dbm.check_connection()
    return HealthOut(
        status="ok" if predictor and db_ok else "degraded",
        model_loaded=predictor is not None,
        database_connected=db_ok,
        model_version=predictor.model_version if predictor else None,
    )


@app.get("/model-info", response_model=ModelInfoOut, tags=["system"],
         dependencies=[Depends(require_api_key)])
def model_info(predictor: FraudPredictor = Depends(get_predictor)) -> dict:
    return predictor.info()


@app.post("/predict", response_model=PredictionOut, tags=["prediction"],
          dependencies=[Depends(require_api_key)])
def predict(txn: TransactionIn, predictor: FraudPredictor = Depends(get_predictor),
            db: Session = Depends(dbm.get_db)) -> PredictionOut:
    transaction_id, record = _to_model_record(txn)
    try:
        result = predictor.predict(record, explain=True)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    stored = _try_store(db, [_to_db_row(txn, transaction_id, record, result)])
    logger.info("predict txn=%s prob=%.4f score=%s level=%s",
                transaction_id, result["fraud_probability"], result["risk_score"], result["risk_level"])
    return _to_response(transaction_id, result, stored)


@app.post("/batch-predict", response_model=BatchOut, tags=["prediction"],
          dependencies=[Depends(require_api_key)])
def batch_predict(body: BatchRequest, predictor: FraudPredictor = Depends(get_predictor),
                  db: Session = Depends(dbm.get_db)) -> BatchOut:
    import pandas as pd

    prepared = [_to_model_record(t) for t in body.transactions]
    frame = pd.DataFrame([rec for _, rec in prepared])
    try:
        results = predictor.predict_dataframe(frame, explain=body.include_explanations)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    rows = [_to_db_row(t, tid, rec, res)
            for t, (tid, rec), res in zip(body.transactions, prepared, results)]
    stored = _try_store(db, rows)
    out = [_to_response(tid, res, stored) for (tid, _), res in zip(prepared, results)]
    return BatchOut(count=len(out), fraud_count=sum(r.prediction for r in out), results=out)


@app.get("/history", response_model=HistoryOut, tags=["analytics"],
         dependencies=[Depends(require_api_key)])
def history(
    date_from: date | None = None,
    date_to: date | None = None,
    risk_level: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH)$"),
    prediction: int | None = Query(None, ge=0, le=1),
    min_amount: float | None = Query(None, ge=0),
    max_amount: float | None = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(dbm.get_db),
) -> HistoryOut:
    total, rows = dbm.query_history(db, date_from, date_to, risk_level, prediction,
                                    min_amount, max_amount, limit, offset)
    return HistoryOut(total=total, items=rows)


@app.get("/stats", response_model=StatsOut, tags=["analytics"],
         dependencies=[Depends(require_api_key)])
def stats(db: Session = Depends(dbm.get_db)) -> dict:
    return dbm.compute_stats(db)
