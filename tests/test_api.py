"""API tests. Uses a real (tiny) trained model and a throw-away SQLite database."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import BAD_TXN, GOOD_TXN  # noqa: E402


@pytest.fixture(scope="module")
def client(trained_dir, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(trained_dir))
    mp.setenv("DATABASE_URL", f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}")
    mp.delenv("API_KEY", raising=False)
    from api.main import app

    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "database_connected": True,
                    "model_version": body["model_version"]}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_type"] in {"XGBoost", "Random Forest"}
    assert body["risk_thresholds"] == {"low_max": 30, "medium_max": 70}
    assert {"precision", "recall", "f1", "pr_auc"} <= body["test_metrics"].keys()
    assert body["global_importance"]


def test_predict_returns_full_contract(client):
    r = client.post("/predict", json=BAD_TXN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["prediction"] in (0, 1)
    assert body["top_increasing_features"] and body["explanation"]
    assert body["stored"] is True and body["transaction_id"].startswith("txn_")


def test_predict_keeps_client_transaction_id(client):
    r = client.post("/predict", json={**GOOD_TXN, "transaction_id": "ABC-123"})
    assert r.json()["transaction_id"] == "ABC-123"


@pytest.mark.parametrize("patch", [
    {"amount": -5}, {"amount": 0}, {"merchant_category": "casino"}, {"payment_channel": "carrier_pigeon"},
    {"txn_count_24h": -1}, {"unexpected_field": 1}, {"timestamp": "not-a-date"}, {"is_new_device": "maybe"},
])
def test_predict_rejects_invalid_input(client, patch):
    assert client.post("/predict", json={**GOOD_TXN, **patch}).status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in GOOD_TXN.items() if k != "amount"}
    assert client.post("/predict", json=payload).status_code == 422


def test_optional_fields_may_be_null(client):
    r = client.post("/predict", json={**GOOD_TXN, "avg_amount_30d": None, "distance_from_home_km": None})
    assert r.status_code == 200


def test_batch_predict(client):
    r = client.post("/batch-predict", json={"transactions": [GOOD_TXN, BAD_TXN]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["results"]) == 2
    assert body["results"][0]["explanation"] is None  # explanations are opt-in for batches
    with_expl = client.post("/batch-predict", json={"transactions": [BAD_TXN], "include_explanations": True})
    assert with_expl.json()["results"][0]["explanation"]


def test_batch_validation(client):
    assert client.post("/batch-predict", json={"transactions": []}).status_code == 422
    assert client.post("/batch-predict", json={"transactions": [GOOD_TXN] * 1001}).status_code == 422


def test_history_and_filters(client):
    client.post("/predict", json={**BAD_TXN, "transaction_id": "HIST-BAD"})
    client.post("/predict", json={**GOOD_TXN, "transaction_id": "HIST-GOOD"})
    everything = client.get("/history", params={"limit": 500}).json()
    ids = {i["transaction_id"] for i in everything["items"]}
    assert {"HIST-BAD", "HIST-GOOD"} <= ids and everything["total"] >= 2

    big = client.get("/history", params={"min_amount": 2000}).json()
    assert all(i["amount"] >= 2000 for i in big["items"]) and big["total"] >= 1
    only_date = client.get("/history", params={"date_from": "2026-03-10", "date_to": "2026-03-10"}).json()
    assert only_date["total"] >= 2
    assert client.get("/history", params={"risk_level": "NOPE"}).status_code == 422


def test_stats(client):
    s = client.get("/stats").json()
    assert s["total_transactions"] >= 1
    assert sum(s["risk_distribution"].values()) == s["total_transactions"]
    assert 0 <= s["fraud_percentage"] <= 100


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.post("/predict", json=GOOD_TXN).status_code == 401
    ok = client.post("/predict", json=GOOD_TXN, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for orchestrators


def test_503_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'x.db'}")
    monkeypatch.delenv("API_KEY", raising=False)
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=GOOD_TXN).status_code == 503
