"""
Prediction Script for House Prices Capstone
Loads the saved Linear Regression model and predicts the price of a new house.
"""

import os
import joblib
import pandas as pd
import numpy as np

def load_trained_model():
    """Load the saved Linear Regression model from the models directory."""
    model_path = os.path.join(os.path.dirname(__file__), '../models/linear_regression_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train and save the model first.")
    
    model = joblib.load(model_path)
    print(f"Model successfully loaded from {model_path}")
    return model

def predict_house_price(house_features):
    """
    Predict the price of a house given its feature dictionary.
    
    Parameters:
    - house_features (dict): Dictionary containing all 13 required model features.
    
    Returns:
    - float: Predicted house price in USD.
    """
    model = load_trained_model()
    
    # Convert input dictionary to DataFrame to match training schema
    input_df = pd.DataFrame([house_features])
    
    # Run prediction
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    # Example new house features matching the training feature structure
    sample_house = {
        'area': 7500,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 1,
        'guestroom': 0,
        'basement': 1,
        'hotwaterheating': 0,
        'airconditioning': 1,
        'parking': 2,
        'prefarea': 1,
        'furnishingstatus_semi-furnished': 1,
        'furnishingstatus_unfurnished': 0
    }
    
    predicted_price = predict_house_price(sample_house)
    print(f"\n--- Prediction Results ---")
    print(f"Estimated House Price: ${predicted_price:,.2f}")