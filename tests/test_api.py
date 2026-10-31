from fastapi.testclient import TestClient
from src.backend_api import app

client = TestClient(app)


def test_health_check():
  response = client.get("/healthz")
  assert response.status_code == 200
  assert response.json()["status"] == "healthy"


def test_predict_endpoint():
  payload = {
      "tenure_months": 4,
      "monthly_charges": 120.0,
      "support_call_frequency": 7,
      "omnichannel_volume": 15,
      "contract_type": 0,
  }
  response = client.post("/predict", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert "churn_prediction" in data
  assert "churn_probability" in data
  assert "risk_category" in data