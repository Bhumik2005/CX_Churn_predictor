import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import MODEL_PATH, SCALER_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CS_Infocomm_MLOps")

app = FastAPI(
    title="CS Infocomm CX Churn Prediction Enterprise API",
    version="2.0.0",
    docs_url="/docs",
)

model, scaler = None, None

try:
  model = joblib.load(MODEL_PATH)
  scaler = joblib.load(SCALER_PATH)
  logger.info("ML Model and Scaler artifacts loaded successfully.")
except Exception as e:
  logger.error(f"Failed to load model artifacts: {e}")


# --- 👇 YAHAN ROOT ENDPOINT ADD KARNA HAI ---
@app.get("/")
def read_root():
  return {
      "message": "Welcome to CS Infocomm CX Churn Prediction Enterprise API",
      "docs_url": "/docs",
  }


class CustomerTelemetrySchema(BaseModel):
  tenure_months: int = Field(..., ge=1, le=72, example=12)
  monthly_charges: float = Field(..., ge=10.0, le=200.0, example=75.5)
  support_call_frequency: int = Field(..., ge=0, example=3)
  omnichannel_volume: int = Field(..., ge=1, example=8)
  contract_type: int = Field(..., ge=0, le=2, example=0)


@app.get("/healthz", status_code=200)
def health_check():
  return {
      "status": "healthy",
      "model_loaded": model is not None,
      "scaler_loaded": scaler is not None,
  }


@app.post("/predict")
def predict_churn(data: CustomerTelemetrySchema):
  if model is None or scaler is None:
    raise HTTPException(
        status_code=500, detail="Model artifacts are not loaded on server."
    )

  try:
    features = np.array([[
        data.tenure_months,
        data.monthly_charges,
        data.support_call_frequency,
        data.omnichannel_volume,
        data.contract_type,
    ]])

    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    probability = float(model.predict_proba(scaled_features)[0][1])

    risk_level = (
        "High Risk"
        if probability > 0.7
        else ("Medium Risk" if probability > 0.4 else "Low Risk")
    )

    logger.info(
        f"Inference successful. Probability: {probability:.4f}, Risk:"
        f" {risk_level}"
    )
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability * 100, 2),
        "risk_category": risk_level,
    }
  except Exception as e:
    logger.error(f"Inference execution error: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))