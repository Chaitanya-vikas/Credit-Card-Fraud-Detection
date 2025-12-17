import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the trained model and scaler
# We use @st.cache to speed up the app so it doesn't reload every time
@st.cache_resource
def load_assets():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. App Title and Description
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("""
This application uses a Machine Learning model (**Random Forest**) to detect whether a credit card transaction is **Fraudulent** or **Legitimate**.
""")

# 3. Sidebar for User Inputs
st.sidebar.header("Input Transaction Details")

# Since we can't ask users for V1-V28 (they are mathematical abstractions), 
# we will simulate inputs or allow basic manual testing.
# For a demo, we will use sliders for the most important features (V14, V4, V11 often correlate high).

def user_input_features():
    # Allow user to input simply Time and Amount
    time = st.sidebar.number_input("Time (Seconds since first transaction)", min_value=0, value=100)
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    
    # For V1-V28, we generate random noise for the demo 
    # (In a real scenario, these come from the bank system)
    data = {'Time': time, 'Amount': amount}
    
    # Add dummy V1-V28 columns with mean values (0) just to satisfy the model format
    for i in range(1, 29):
        data[f'V{i}'] = 0.0 # Default to average
        
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Display User Input
st.subheader("Transaction Details")
st.write(input_df[['Time', 'Amount']]) # Show only understandable cols

# 5. Prediction Logic
if st.button("Analyze Transaction"):
    # Preprocess: Scale the Time and Amount just like we did in training
    # Note: We must reshape because scaler expects 2D array
    input_df['scaled_amount'] = scaler.fit_transform(input_df['Amount'].values.reshape(-1,1))
    input_df['scaled_time'] = scaler.fit_transform(input_df['Time'].values.reshape(-1,1))
    
    # Drop original Time/Amount to match training columns
    final_input = input_df.drop(['Time', 'Amount'], axis=1)
    
    # Reorder columns: scaled_amount, scaled_time, V1...V28
    # (Ensure this matches exactly the order you trained on!)
    # Ideally, we load the column names from training, but for now we reconstruct:
    cols = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
    # This part handles column ordering dynamically if needed, 
    # but for this demo, we assume the V columns are 0.
    
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)

    st.subheader("Risk Assessment")
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUD DETECTED! (Risk Probability: {probability[0][1]:.2%})")
    else:
        st.success(f"✅ Transaction Legitimate (Safety Probability: {probability[0][0]:.2%})")