import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_artifact.pkl")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_telemetry.csv")