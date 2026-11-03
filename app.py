"""
Streamlit Web App for House Price Prediction Capstone
Provides an interactive user interface to test the trained Linear Regression model.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models/linear_regression_model.pkl')
    return joblib.load(model_path)

model = load_model()

# App Header
st.title("🏡 House Price Prediction App")
st.write("Enter the property details below to estimate the predicted market value using your trained Machine Learning model.")

st.markdown("---")

# User Input Controls in a Form/Layout
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq. ft.)", min_value=1500, max_value=20000, value=7500, step=100)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2)
    stories = st.slider("Stories", min_value=1, max_value=4, value=2)
    parking = st.slider("Parking Spaces", min_value=0, max_value=4, value=2)

with col2:
    mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
    guestroom = st.selectbox("Guest Room", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Convert user selections into model feature format
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': 1 if mainroad == "Yes" else 0,
    'guestroom': 1 if guestroom == "Yes" else 0,
    'basement': 1 if basement == "Yes" else 0,
    'hotwaterheating': 1 if hotwaterheating == "Yes" else 0,
    'airconditioning': 1 if airconditioning == "Yes" else 0,
    'parking': parking,
    'prefarea': 1 if prefarea == "Yes" else 0,
    'furnishingstatus_semi-furnished': 1 if furnishing == "Semi-Furnished" else 0,
    'furnishingstatus_unfurnished': 1 if furnishing == "Unfurnished" else 0
}

st.markdown("---")

# Predict Button & Output
if st.button("Predict House Price 🚀", type="primary"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Property Price: ${prediction:,.2f}")