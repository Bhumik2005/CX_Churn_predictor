import os
import numpy as np
import pandas as pd
from src.config import DATA_DIR, RAW_DATA_PATH


def generate_telemetry_dataset(num_samples=5000, random_state=42):
  np.random.seed(random_state)

  tenure = np.random.randint(1, 72, size=num_samples)
  monthly_charges = np.random.uniform(20.0, 150.0, size=num_samples)
  support_calls = np.random.poisson(lam=1.5, size=num_samples)
  omnichannel_interactions = np.random.randint(1, 20, size=num_samples)
  contract_type = np.random.choice(
      [0, 1, 2], size=num_samples, p=[0.5, 0.3, 0.2]
  )

  churn_prob = (
      0.03 * (72 - tenure)
      + 0.01 * monthly_charges
      + 0.15 * support_calls
      - 0.05 * contract_type
      + np.random.normal(0, 0.2, size=num_samples)
  )

  churn_prob = 1 / (1 + np.exp(-churn_prob + 2))
  churn = (churn_prob > 0.65).astype(int)

  df = pd.DataFrame({
      "tenure_months": tenure,
      "monthly_charges": monthly_charges,
      "support_call_frequency": support_calls,
      "omnichannel_volume": omnichannel_interactions,
      "contract_type": contract_type,
      "churn": churn,
  })

  os.makedirs(DATA_DIR, exist_ok=True)
  df.to_csv(RAW_DATA_PATH, index=False)
  print(f"Telemetry dataset generated successfully at: {RAW_DATA_PATH}")
  return df


if __name__ == "__main__":
  generate_telemetry_dataset()