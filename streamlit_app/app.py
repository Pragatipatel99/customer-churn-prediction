import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing pipeline
model = joblib.load("../models/churn_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.sidebar.header("Customer Information")

# Customer inputs
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# Prediction button
if st.button("Predict Churn"):

    # Create engineered features
    if tenure <= 12:
        tenure_group = "0-12 months"
    elif tenure <= 48:
        tenure_group = "13-48 months"
    else:
        tenure_group = "49+ months"

    if monthly_charges <= 40:
        monthly_charge_group = "Low"
    elif monthly_charges <= 80:
        monthly_charge_group = "Medium"
    else:
        monthly_charge_group = "High"

    has_dependents = 1 if dependents == "Yes" else 0

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": ["No phone service" if phone_service == "No" else "No"],
        "InternetService": [internet_service],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "tenure_group": [tenure_group],
        "monthly_charge_group": [monthly_charge_group],
        "has_dependents": [has_dependents],
        "total_services": [0]
    })

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )