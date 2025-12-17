import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# 2. App Title
st.title("🛡️ Financial Fraud Detection System")
st.write("Enter transaction details to check if they are **Legitimate** or **Fraudulent**.")

# Check if model loaded correctly
if model is None:
    st.error("🚨 Error: Model files not found. Please make sure 'fraud_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Transaction Details")

# User inputs
time_val = st.sidebar.number_input("Time (Secs since midnight)", min_value=0, value=40000)
amount_val = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.0)

# 4. Predict Button
if st.button("Check Transaction"):
    
    # --- DATA PREPARATION ---
    
    # 1. Create a dictionary with default values for V1-V28
    input_data = {f'V{i}': 0.0 for i in range(1, 29)}
    
    # 2. Handle Scaling
    # We use a try-except block to handle different scaler types safeley
    try:
        # Reshape to 2D array as required by sklearn
        scaled_amount = scaler.transform([[amount_val]])[0][0]
        scaled_time = scaler.transform([[time_val]])[0][0]
    except:
        # Fallback if scaler fails: approximate robust scaling manually
        # (This prevents app crash if the scaler.pkl is incompatible)
        scaled_amount = (amount_val - 88.0) / 71.0
        scaled_time = (time_val - 84000.0) / 85000.0
    
    # 3. Add scaled values to the input dict
    input_data['scaled_amount'] = scaled_amount
    input_data['scaled_time'] = scaled_time
    
    # 4. Convert to DataFrame
    raw_df = pd.DataFrame([input_data])
    
    # --- THE FIX: AUTO-ALIGN COLUMNS ---
    # We ask the model: "What columns do you want?" and we reorder our data to match.
    if hasattr(model, 'feature_names_in_'):
        try:
            final_df = raw_df[model.feature_names_in_]
        except KeyError as e:
            st.error(f"Column Mismatch! The model expects: {list(model.feature_names_in_)}")
            st.stop()
    else:
        final_df = raw_df  # Fallback for older scikit-learn versions

    # --- PREDICTION ---
    prediction = model.predict(final_df)
    probability = model.predict_proba(final_df)
    
    st.subheader("Risk Assessment")
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUD ALERT! (Risk Probability: {probability[0][1]:.2%})")
        st.write("**Recommendation:** Flag this transaction for manual review.")
    else:
        st.success(f"✅ Transaction Legitimate (Safety Probability: {probability[0][0]:.2%})")