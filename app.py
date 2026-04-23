import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

# --- Page Config ---
st.set_page_config(page_title="Credit Fraud Detector", layout="centered")

# --- Load Model and Scaler ---
@st.cache_resource
def load_resources():
    # Load the XGBoost model
    model = xgb.XGBClassifier()
    model.load_model("my_model.json")
    
    # Recreate the scaler using the original data logic
    # (Since the scaler object was not saved separately, we fit it here)
    df = pd.read_csv("credit_fraud.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Cleaning 'age' as per your notebook logic
    df["age"] = df["age"].astype(str).str.replace("_err", "")
    df["age"] = pd.to_numeric(df["age"], errors='coerce').fillna(31).astype(int)
    
    X = df.drop("is_fraud", axis=1)
    
    scaler = StandardScaler()
    scaler.fit(X) # Fitting on the cleaned features
    
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = load_resources()

# --- UI Layout ---
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Enter the transaction details below to check the probability of a fraudulent transaction.
""")

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        txn_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1500.0)
        acc_balance = st.number_input("Account Balance ($)", min_value=0.0, value=25000.0)
        num_txns = st.number_input("Transactions Today", min_value=0, value=5)
        txn_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 12)
        
    with col2:
        foreign_txn = st.selectbox("Foreign Transaction?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prev_fraud = st.selectbox("Previous Fraud Flag?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        m_dist = st.number_input("Merchant Distance (km)", min_value=0.0, value=10.5)
        m_risk = st.slider("Merchant Risk Score", 0.0, 10.0, 5.0)

    submit = st.form_submit_button("Detect Fraud")

# --- Prediction Logic ---
if submit:
    # 1. Create a DataFrame for the input
    input_data = pd.DataFrame([[
        age, txn_amount, acc_balance, num_txns, 
        foreign_txn, txn_hour, prev_fraud, m_dist, m_risk
    ]], columns=feature_names)
    
    # 2. Scale the input
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 4. Display Result
    st.divider()
    if prediction == 1:
        st.error(f"🚨 **Warning: Potential Fraud Detected!**")
        st.metric("Fraud Probability", f"{probability*100:.2f}%")
    else:
        st.success(f"✅ **Transaction Appears Safe.**")
        st.metric("Fraud Probability", f"{probability*100:.2f}%")