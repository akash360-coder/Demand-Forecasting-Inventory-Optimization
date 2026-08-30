from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_endpoint_returns_payload():
    response = client.get("/api/v1/forecast", params={"product_id": "P101", "store_id": 1, "forecast_horizon": 7})
    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert "points" in payload
    assert len(payload["points"]) == 7


def test_dashboard_endpoint_returns_summary():
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "demand_today" in payload
    assert "risk_breakdown" in payload
