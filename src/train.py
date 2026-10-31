import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.config import MODEL_DIR, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH


def train_churn_model():
  if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        "Raw telemetry dataset not found. Run data_generator.py first."
    )

  df = pd.read_csv(RAW_DATA_PATH)
  
  # Explicit feature selection matching schema exactly
  feature_cols = [
      "tenure_months",
      "monthly_charges",
      "support_call_frequency",
      "omnichannel_volume",
      "contract_type",
  ]
  
  X = df[feature_cols]
  y = df["churn"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  smote = SMOTE(random_state=42)
  X_train_balanced, y_train_balanced = smote.fit_resample(
      X_train_scaled, y_train
  )

  model = XGBClassifier(
      n_estimators=100,
      learning_rate=0.1,
      max_depth=5,
      random_state=42,
      eval_metric="logloss",
  )
  model.fit(X_train_balanced, y_train_balanced)

  os.makedirs(MODEL_DIR, exist_ok=True)
  joblib.dump(model, MODEL_PATH)
  joblib.dump(scaler, SCALER_PATH)
  print("XGBoost model and scaler artifacts re-trained & serialized successfully!")


if __name__ == "__main__":
  train_churn_model()