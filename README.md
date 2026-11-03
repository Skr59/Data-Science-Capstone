# 🏡 House Prices Prediction & Data Science Capstone

An end-to-end machine learning pipeline built to analyze real estate data, perform exploratory data analysis (EDA), train and serialize regression models, and generate production-ready property valuations.

This project includes a working Streamlit app for predicting home prices from user-entered property features and a reusable Python prediction script that loads the trained model.

## Features
- Exploratory analysis and preprocessing notebook
- Trained linear regression model for house price prediction
- Random forest model for benchmarking/comparison
- Streamlit dashboard for interactive house price estimation
- Reusable prediction utility in `src/predict.py`

## Local setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project structure

```text
Data-Science-Capstone/
│
├── app.py                          # Streamlit app UI and prediction logic
├── data/
│   └── house_prices.csv           # Raw dataset used for modeling
├── models/
│   ├── linear_regression_model.pkl # Serialized baseline regression model
│   └── random_forest_model.pkl    # Serialized benchmark model
├── notebooks/
│   └── 01-eda.ipynb              # EDA, preprocessing, and model training workflow
├── src/
│   └── predict.py                 # Prediction helper compatible with the saved model
├── requirements.txt               # Dependency list for local setup
├── .gitignore                     # Common environment and artifact exclusions
├── README.md                      # Project documentation
└── .venv/                         # Local Python environment
```

## Prediction example

The trained model expects these input features:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus_semi-furnished`
- `furnishingstatus_unfurnished`

You can also evaluate the model programmatically:

```python
from src.predict import predict_house_price

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
    'furnishingstatus_unfurnished': 0,
}

prediction = predict_house_price(sample_house)
print(f"Estimated price: ${prediction:,.2f}")
```

## Notes
- The app is designed for a local environment and runs with the Python dependencies listed in `requirements.txt`.
- The saved model file is included in the repository so the app can be used immediately after installing dependencies.