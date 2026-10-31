# 🌐 Enterprise Omnichannel CX Churn Prediction MLOps Platform

An end-to-end, production-ready MLOps platform designed and developed during an intensive technical internship. This system predicts customer churn probabilities for omnichannel businesses using an optimized XGBoost classifier, serves real-time predictions via a robust FastAPI backend, and provides an executive monitoring interface through a custom-styled Streamlit dashboard.

---

## 🚀 Project Overview & Highlights
* **End-to-End MLOps Pipeline:** Covers synthetic telemetry generation, data preprocessing with SMOTE (handling class imbalance), model training, automated testing, API serving, and executive UI visualization.
* **Backend Microservice:** Built using **FastAPI** and **Pydantic** for rigorous schema validation, featuring `/healthz`, root status, and `/predict` endpoints.
* **Rigorous Automated Testing:** Integrated with `pytest` ensuring 100% unit test pass rates and high API reliability.
* **Enterprise UI/UX:** A sleek, light-themed executive dashboard built with **Streamlit** and custom CSS, featuring real-time risk categorization (Low, Medium, High Risk) and interactive telemetry sliders.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.10 (isolated inside virtual environments `venv`)
* **Machine Learning:** Scikit-Learn, XGBoost, SMOTE (Imbalanced-Learn)
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend / UI:** Streamlit, Custom CSS (Enterprise Light Theme)
* **Testing:** Pytest
* **Containerization & Deployment:** Docker, Docker-Compose (Structure ready)
* **Version Control:** Git & GitHub

---

## 📂 Project Directory Structure
```text
CX_Churn_Predictor/
│
├── data/                  # Generated synthetic customer telemetry datasets
├── models/                # Serialized XGBoost model artifacts (.pkl)
├── src/
│   ├── __init__.py
│   ├── data_generator.py  # Synthetic customer telemetry data pipeline
│   ├── train.py           # Model training script with SMOTE & XGBoost
│   └── backend_api.py     # FastAPI application & Pydantic schema validation
├── tests/
│   └── test_api.py        # Automated unit tests using pytest
├── dashboard.py           # Streamlit executive UI workspace
├── requirements.txt       # Project Python dependencies
├── DockerFile             # Container configuration file
└── docker-compose.yaml    # Multi-container orchestration setup
```
⚙️ Local Setup & Installation Guide
Follow these steps to set up and run the platform locally on your machine:

1. Clone the Repository
   ```
   git clone [https://github.com/Bhumik2005/CX_Churn_predictor.git](https://github.com/Bhumik2005/CX_Churn_predictor.git)
   cd CX_Churn_Predictor
   ```
2. Set Up Virtual Environment
```
python -m venv venv
.\venv\Scripts\Activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the FastAPI Backend Server
Start the backend inference engine on port 8000:
```
uvicorn src.backend_api:app --reload --port 8000
```
(You can verify the API documentation at http://127.0.0.1:8000/docs)

5. Launch the Streamlit Executive Dashboard
Open a separate terminal window, activate your virtual environment, and run:
```
.\venv\Scripts\Activate
streamlit run dashboard.py
```
🧪 Running Automated Unit Tests
To execute the test suite and verify system health:
```
pytest -v
```

📈 Future Roadmap & Enhancements
- Data Drift Monitoring: Integration with Evidently AI for real-time production drift detection.

- Persistent Database Integration: Connecting PostgreSQL for secure transaction and prediction logging.

- Explainable AI (XAI): Implementing SHAP values on the Streamlit dashboard to interpret individual customer risk factors.

- Cloud Deployment: Containerizing via Docker and deploying to AWS/Render cloud infrastructure.

  
