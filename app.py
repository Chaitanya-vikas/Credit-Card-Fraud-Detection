import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Error: .pkl files not found. Make sure 'fraud_model.pkl' and 'scaler.pkl' are in the same folder.")
    st.stop()

# 2. App Title
st.title("🛡️ Financial Fraud Detection System")
st.write("Enter transaction details to check if they are **Legitimate** or **Fraudulent**.")

# 3. Sidebar Inputs
st.sidebar.header("Transaction Details")

# User inputs Time and Amount
# We set default values to match a "Normal" transaction range
time_val = st.sidebar.number_input("Time (Secs since midnight)", min_value=0, value=40000)
amount_val = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.0)

# 4. Predict Button
if st.button("Check Transaction"):
    
    # --- DATA PREPARATION ---
    # The model expects 30 columns in a specific order:
    # [V1, V2, ..., V28, scaled_amount, scaled_time]
    
    # 1. Create a dictionary with all features set to 0 (average)
    input_data = {f'V{i}': 0.0 for i in range(1, 29)}
    
    # 2. Manually scale the user inputs using the saved scaler
    # Note: We create a temporary dataframe just for the scaler to avoid shape errors
    # The scaler expects [[Amount], [Time]] or similar depending on how it was fitted. 
    # To be safe and robust, we manually approximate based on the input values 
    # OR we use the scaler if we are sure of the shape.
    
    # robust_scaler usually subtracts Median and divides by IQR.
    # Let's use the loaded scaler to be 100% accurate.
    # We must reshape inputs to 2D arrays: [[value]]
    scaled_amount = scaler.transform([[amount_val]])[0][0] if hasattr(scaler, 'transform') else amount_val
    scaled_time = scaler.transform([[time_val]])[0][0] if hasattr(scaler, 'transform') else time_val
    
    # 3. Add the scaled values to our input dictionary
    input_data['scaled_amount'] = scaled_amount
    input_data['scaled_time'] = scaled_time
    
    # 4. Convert to DataFrame (This fixes the "Feature Names" error!)
    # We enforce the column order: V1...V28, scaled_amount, scaled_time
    columns_order = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
    final_input_df = pd.DataFrame([input_data], columns=columns_order)

    # --- PREDICTION ---
    prediction = model.predict(final_input_df)
    probability = model.predict_proba(final_input_df)
    
    st.subheader("Risk Assessment")
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUD ALERT! (Risk: {probability[0][1]:.2%})")
        st.write("**Recommendation:** Flag this transaction for manual review.")
    else:
        st.success(f"✅ Transaction Legitimate (Safety: {probability[0][0]:.2%})")