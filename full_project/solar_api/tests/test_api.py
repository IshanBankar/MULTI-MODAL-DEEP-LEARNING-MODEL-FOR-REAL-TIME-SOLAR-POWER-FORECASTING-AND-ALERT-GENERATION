from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["demo_samples_available"] > 0


def test_predict_sample():
    r = client.get("/predict/sample/0")
    assert r.status_code == 200
    body = r.json()
    assert "predicted_power_kw" in body
    assert "actual_power_kw" in body
    assert body["predicted_power_kw"] >= 0
    assert body["latency_ms"] > 0


def test_predict_sample_out_of_range():
    r = client.get("/predict/sample/999999")
    assert r.status_code == 404
