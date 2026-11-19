from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "MedInc": 8.0,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "estimated_price_usd" in result
    assert isinstance(result["prediction"], float)
    assert result["estimated_price_usd"] > 0


def test_prediction_rejects_invalid_input():
    payload = {
        "MedInc": 8.0,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000,
        "AveOccup": 0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "housing_prediction_requests" in response.text
