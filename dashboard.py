import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="CX Churn Prediction | Enterprise MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Blue & White Light Theme with Clear Labels
st.markdown(
    """
    <style>
    /* Main background & font styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling & Label Visibility */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Header Typography */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Card Containers */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
    }
    
    /* Risk Badge Styling */
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        border: 1px solid #f87171;
    }
    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        border: 1px solid #fbbf24;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        border: 1px solid #4ade80;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Inputs
st.sidebar.markdown(
    "### ⚙️ Customer Telemetry Parameters", unsafe_allow_html=True
)
st.sidebar.markdown(
    "Adjust customer metrics below to evaluate churn probability in real-time."
)

tenure_months = st.sidebar.slider(
    "Tenure (Months)", min_value=1, max_value=72, value=12
)
monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)", min_value=10.0, max_value=200.0, value=75.5
)
support_call_frequency = st.sidebar.slider(
    "Support Call Frequency", min_value=0, max_value=15, value=2
)
omnichannel_volume = st.sidebar.slider(
    "Omnichannel Interactions", min_value=1, max_value=30, value=6
)

contract_type_map = {"Monthly-to-Monthly": 0, "One Year": 1, "Two Year": 2}
contract_selection = st.sidebar.selectbox(
    "Contract Type", list(contract_type_map.keys())
)
contract_type = contract_type_map[contract_selection]

# Main Interface Layout
st.markdown("# 🌐 Enterprise Omnichannel CX Churn Prediction")
st.markdown(
    "Real-time customer retention monitoring powered by MLOps and XGBoost"
    " inference engine."
)
st.markdown("---")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
  st.markdown("### 📋 Input Profile Summary")
  st.markdown(
      f"""
    <div class="metric-card">
        <ul>
            <li><b>Tenure:</b> {tenure_months} months</li>
            <li><b>Monthly Charges:</b> ${monthly_charges:.2f}</li>
            <li><b>Support Calls:</b> {support_call_frequency}</li>
            <li><b>Interactions:</b> {omnichannel_volume}</li>
            <li><b>Contract:</b> {contract_selection}</li>
        </ul>
    </div>
    """,
      unsafe_allow_html=True,
  )

  evaluate_btn = st.button("Evaluate Churn Risk")

with col2:
  st.markdown("### 📈 Real-Time Prediction & Risk Analysis")

  if evaluate_btn:
    payload = {
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "support_call_frequency": support_call_frequency,
        "omnichannel_volume": omnichannel_volume,
        "contract_type": contract_type,
    }

    try:
      response = requests.post("http://127.0.0.1:8000/predict", json=payload)
      if response.status_code == 200:
        res_data = response.json()
        prob = res_data["churn_probability"]
        risk = res_data["risk_category"]

        badge_class = (
            "risk-high"
            if risk == "High Risk"
            else ("risk-medium" if risk == "Medium Risk" else "risk-low")
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Inference Results</h4>
                <p><b>Churn Probability:</b> {prob}%</p>
                <div class="{badge_class}">Risk Level: {risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
      else:
        st.error(
            "API Error: Failed to fetch prediction from backend service."
        )
    except requests.exceptions.ConnectionError:
      st.error(
          "Connection Error: Make sure FastAPI backend server is running on port"
          " 8000 (`uvicorn src.backend_api:app --reload --port 8000`)."
      )
  else:
    st.markdown(
        """
        <div class="metric-card" style="color: #64748b; text-align: center; padding: 40px;">
            <h4>Awaiting Evaluation</h4>
            <p>Configure parameters on the sidebar and click <b>'Evaluate Churn Risk'</b> to trigger model inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )