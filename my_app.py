import streamlit as st
import requests
import pandas as pd

data = pd.read_csv('C:\\Users\\shahi\\OneDrive\\Desktop\\App\\MLE-Assignment.csv')
ds = pd.DataFrame(data)

# Streamlit UI
st.title("🌽 Corn Mycotoxin (DON) Prediction 🚜")
st.write("Enter spectral reflectance values to predict DON concentration.")

# Number of features
num_features = 448 # Change this based on your dataset
num_columns_per_row = 5  # You can adjust this based on the UI size (max 8 columns for better readability)

# Create empty list to store the input values
input_values = []

# Display inputs in tabular format using columns
st.write("Enter Spectral Reflectance Values:")

# Loop to create rows of inputs with columns
for i in range(0, num_features, num_columns_per_row):
    # Create a row with 'num_columns_per_row' columns
    cols = st.columns(num_columns_per_row)
    for j, col in enumerate(cols):
        if i + j < num_features:
            value = col.number_input(f"Wavelength {i + j + 1} Reflectance", min_value=0.0, max_value=1.0, step=0.01, key=f"wavelength_{i + j}")
            input_values.append(value)

# Convert input values to a list for API request
input_data = {"reflectance_values": input_values}

# API Endpoint
API_URL = "http://127.0.0.1:8080/predict/"

# Prediction Button
if st.button("🔍 Predict DON Concentration"):
    try:
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            prediction = response.json().get("predicted_don_concentration")
            st.success(f"🌟 Predicted DON Concentration: {prediction:.2f} ppb")
        else:
            st.error("⚠️ Error in prediction. Check API.")
    
    except requests.exceptions.ConnectionError:
        st.error("🚨 Unable to connect to the FastAPI server. Make sure it's running!")
