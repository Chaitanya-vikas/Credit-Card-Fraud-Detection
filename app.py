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
st.write("Adjust the transaction details to see if the model flags it as **Fraud**.")

if model is None:
    st.error("🚨 Error: Model files not found.")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Transaction Details")

# Standard Inputs
time_val = st.sidebar.number_input("Time (Secs)", min_value=0, value=40000)
amount_val = st.sidebar.number_input("Amount ($)", min_value=0.0, value=120.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🕵️ Fraud Signal Injection")
st.sidebar.caption("Adjust these hidden features (V4 & V14) to simulate suspicious behavior.")

# SLIDERS: This is the fix. Allow user to change the "V" features manually.
# Normal range is usually -3 to 3. Fraud is often > 5 or < -5.
v4_val = st.sidebar.slider("V4 (High = Suspicious)", min_value=-10.0, max_value=15.0, value=0.0)
v14_val = st.sidebar.slider("V14 (Low = Suspicious)", min_value=-20.0, max_value=5.0, value=0.0)

# 4. Predict Button
if st.button("Analyze Transaction"):
    
    # --- DATA PREPARATION ---
    # 1. Start with average values (0.0) for all 28 features
    input_data = {f'V{i}': 0.0 for i in range(1, 29)}
    
    # 2. Overwrite with User Inputs from Sliders
    input_data['V4'] = v4_val
    input_data['V14'] = v14_val
    
    # 3. Handle Scaling for Time/Amount
    try:
        scaled_amount = scaler.transform([[amount_val]])[0][0]
        scaled_time = scaler.transform([[time_val]])[0][0]
    except:
        # Fallback manual scaling
        scaled_amount = (amount_val - 88.0) / 71.0
        scaled_time = (time_val - 84000.0) / 85000.0
    
    input_data['scaled_amount'] = scaled_amount
    input_data['scaled_time'] = scaled_time
    
    # 4. Convert to DataFrame
    raw_df = pd.DataFrame([input_data])
    
    # Auto-align columns to match model training
    if hasattr(model, 'feature_names_in_'):
        try:
            final_df = raw_df[model.feature_names_in_]
        except KeyError:
            st.error("Model feature mismatch.")
            st.stop()
    else:
        final_df = raw_df 

    # --- PREDICTION ---
    prediction = model.predict(final_df)
    probability = model.predict_proba(final_df)
    
    st.subheader("Analysis Result")
    
    # If the probability of fraud (index 1) is greater than 10% (0.10)
    if probability[0][1] > 0.10: 
    st.error(f"🚨 FRAUD DETECTED! (Risk: {probability[0][1]:.2%})")

    else:
        st.success(f"✅ Legitimate Transaction (Safety: {probability[0][0]:.2%})")
        st.info("Tip: Try moving the 'V14' slider to -10 or lower to trigger a fraud alert.")