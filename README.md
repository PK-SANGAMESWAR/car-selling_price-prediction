# ML Case Studies

This repository contains two mini-projects:

- car-selling_price-prediction: Predict car resale prices using Linear, Ridge, and Lasso regression with preprocessing pipelines.
- diabetes_prediction_with_knn: Classify diabetes presence using a KNN model with k-optimization.

## Quick Start

### Prerequisites
- Python 3.9+
- Git

### Setup (recommended: virtual environment)
`powershell
# Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r car-selling_price-prediction/requirements.txt
pip install -r diabetes_prediction_with_knn/requirements.txt
`

Alternatively, with uv:
`powershell
uv sync
`

## Projects

### Car Selling Price Prediction
- Notebook: car-selling_price-prediction/carprediction.ipynb
- App entry: car-selling_price-prediction/main.py
- Data: car-selling_price-prediction/Cleaned_Car_data.csv, quikr_car.csv

Run the app:
`powershell
# inside venv
python car-selling_price-prediction/main.py
`

### Diabetes Prediction with KNN
- Notebook: diabetes_prediction_with_knn/diabetes_prediction_with_knn.ipynb
- App entry: diabetes_prediction_with_knn/main.py

Run the app:
`powershell
# inside venv
python diabetes_prediction_with_knn/main.py
`

## Notes
- Large artifacts like .pkl are included for convenience; training steps are in the notebooks.
- See notebooks for EDA, preprocessing, model training, and evaluation visuals.

