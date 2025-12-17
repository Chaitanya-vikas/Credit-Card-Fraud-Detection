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
st.write("This app predicts the probability of a transaction being fraudulent.")

if model is None:
    st.error("🚨 Error: Model files not found.")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Transaction Simulation")
scenario = st.sidebar.selectbox("Select Scenario:", ["Legitimate Transaction", "Suspicious Transaction (Fraud Test)"])

# Default legitimate values
time_val = 40000
amount_val = 120.0
v4_val = 0.0
v14_val = 0.0

# If user chooses "Fraud Test", we pre-fill with data from an actual fraud case in the dataset
if scenario == "Suspicious Transaction (Fraud Test)":
    st.sidebar.warning("⚠️ Simulating specific fraud pattern (High V4, Low V14)")
    time_val = 406.0         # Example fraud time
    amount_val = 500.0       # Higher amount
    v4_val = 6.0             # High outlier (typical of fraud)
    v14_val = -8.0           # Low outlier (typical of fraud)
else:
    # Allow manual adjustment for normal testing
    amount_val = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.0)
    time_val = st.sidebar.number_input("Time (Secs)", min_value=0, value=40000)

# 4. Predict Button
if st.button("Analyze Transaction"):
    
    # --- DATA PREPARATION ---
    # 1. Create a dictionary with default values (0.0 represents the average)
    input_data = {f'V{i}': 0.0 for i in range(1, 29)}
    
    # 2. Inject the "Fraud Signals" (V4 and V14 are the most important features in this dataset)
    input_data['V4'] = v4_val
    input_data['V14'] = v14_val
    
    # 3. Handle Scaling
    try:
        scaled_amount = scaler.transform([[amount_val]])[0][0]
        scaled_time = scaler.transform([[time_val]])[0][0]
    except:
        scaled_amount = (amount_val - 88.0) / 71.0
        scaled_time = (time_val - 84000.0) / 85000.0
    
    input_data['scaled_amount'] = scaled_amount
    input_data['scaled_time'] = scaled_time
    
    # 4. Convert to DataFrame and align columns
    raw_df = pd.DataFrame([input_data])
    
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
    
    # Display the inputs used so the user understands WHY it happened
    st.caption(f"Inputs used: Amount=${amount_val}, V4={v4_val}, V14={v14_val}")
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUD DETECTED! (Risk: {probability[0][1]:.2%})")
        st.markdown("**Reasoning:** The model detected abnormal patterns in `V14` and `V4`, typical of stolen card usage.")
    else:
        st.success(f"✅ Transaction Legitimate (Safety: {probability[0][0]:.2%})")