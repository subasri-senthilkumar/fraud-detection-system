"""PostgreSQL persistence with SQLAlchemy 2.x.
 
Configuration comes only from environment variables - no credentials in code:
 
    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db      (takes priority)
or  POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
 
If neither is set the app falls back to a local SQLite file so that development
and the unit tests work without a database server. Production must use Postgres.
"""
from __future__ import annotations
 
import logging
import os
from datetime import date, datetime, timezone
from typing import Iterator
 
from sqlalchemy import (
    JSON, Date, DateTime, Float, Integer, String, case, cast, create_engine, func, select, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
 
logger = logging.getLogger(__name__)
 
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
 
 
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
 
 
class Base(DeclarativeBase):
    pass
 
 
class PredictionRecord(Base):
    __tablename__ = "predictions"
 
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    transaction_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    amount: Mapped[float] = mapped_column(Float, index=True)
    input_data: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(8), index=True)
    prediction: Mapped[int] = mapped_column(Integer, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
 
 
# --------------------------------------------------------------------------- #
def build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        # Some PaaS providers still hand out the legacy "postgres://" scheme.
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        return url
 
    host = os.getenv("POSTGRES_HOST")
    if host:
        missing = [v for v in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB") if not os.getenv(v)]
        if missing:
            raise RuntimeError(f"POSTGRES_HOST is set but {missing} are not")
        return URL.create(
            "postgresql+psycopg2",
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=host,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.environ["POSTGRES_DB"],
        ).render_as_string(hide_password=False)
 
    logger.warning("No DATABASE_URL / POSTGRES_* configured - using local SQLite (dev only).")
    return "sqlite:///./fraud_local.db"
 
 
def init_engine(url: str | None = None) -> Engine:
    """Create the engine + session factory and the tables (idempotent)."""
    global _engine, _SessionLocal
    url = url or build_database_url()
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    _engine = create_engine(url, **kwargs)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    logger.info("Database ready (%s)", _engine.url.render_as_string(hide_password=True))
    return _engine
 
 
def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
 
def check_connection() -> bool:
    if _engine is None:
        return False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database health check failed")
        return False
 
 
# --------------------------------------------------------------------------- #
# Repository functions
# --------------------------------------------------------------------------- #
def save_predictions(db: Session, records: list[PredictionRecord]) -> None:
    db.add_all(records)
    db.commit()
 
 
def _day(db: Session, expr):
    """Date part of a datetime expression, portable across databases.
 
    PostgreSQL: CAST(x AS DATE). SQLite has no real DATE type - CAST(x AS DATE)
    returns a number (e.g. 2026) - so there we use its date() function instead.
    """
    if db.get_bind().dialect.name == "sqlite":
        return func.date(expr, type_=Date)
    return cast(expr, Date)
 
 
def query_history(
    db: Session,
    date_from: date | None = None,
    date_to: date | None = None,
    risk_level: str | None = None,
    prediction: int | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[PredictionRecord]]:
    stmt = select(PredictionRecord)
    # Filter by the time the transaction happened (fall back to scoring time).
    when = func.coalesce(PredictionRecord.transaction_time, PredictionRecord.timestamp)
    if date_from:
        stmt = stmt.where(_day(db, when) >= date_from)
    if date_to:
        stmt = stmt.where(_day(db, when) <= date_to)
    if risk_level:
        stmt = stmt.where(PredictionRecord.risk_level == risk_level)
    if prediction is not None:
        stmt = stmt.where(PredictionRecord.prediction == prediction)
    if min_amount is not None:
        stmt = stmt.where(PredictionRecord.amount >= min_amount)
    if max_amount is not None:
        stmt = stmt.where(PredictionRecord.amount <= max_amount)
 
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(when.desc(), PredictionRecord.id.desc())
                      .limit(limit).offset(offset)).all()
    return total, list(rows)
 
 
def compute_stats(db: Session, days: int = 90) -> dict:
    total, fraud, avg_score, high = db.execute(select(
        func.count(PredictionRecord.id),
        func.coalesce(func.sum(PredictionRecord.prediction), 0),
        func.coalesce(func.avg(PredictionRecord.risk_score), 0.0),
        func.coalesce(func.sum(case((PredictionRecord.risk_level == "HIGH", 1), else_=0)), 0),
    )).one()
 
    dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for level, n in db.execute(select(PredictionRecord.risk_level, func.count())
                               .group_by(PredictionRecord.risk_level)).all():
        dist[level] = int(n)
 
    day = _day(db, func.coalesce(PredictionRecord.transaction_time, PredictionRecord.timestamp))
    trend_rows = db.execute(
        select(day.label("day"), func.count(), func.sum(PredictionRecord.prediction))
        .group_by(day).order_by(day.desc()).limit(days)
    ).all()
    trend = [{"day": d if isinstance(d, date) else date.fromisoformat(str(d)),
              "total": int(t), "fraud": int(f or 0)} for d, t, f in reversed(trend_rows)]
 
    total, fraud = int(total), int(fraud)
    return {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "fraud_percentage": round(100 * fraud / total, 3) if total else 0.0,
        "average_risk_score": round(float(avg_score), 2),
        "high_risk_transactions": int(high),
        "risk_distribution": dist,
        "daily_trend": trend,
    }
 
