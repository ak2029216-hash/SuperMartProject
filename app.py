from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the newly trained Production XGBoost model file
try:
    with open("xgb_sales_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("[System Error] Serialized production asset 'xgb_sales_model.pkl' not found.")

@app.route('/')
def home():
    # Pass metadata parameters directly to your index dashboard container
    performance_metadata = {
        "champion_model": "XGBoost Regressor",
        "validation_rmse": 1041.50,
        "validation_mae": 714.22,
        "dataset_outlets": 10,
        "dataset_items": 1500,
        "developer_name": "Laiba Noor"
    }
    return render_template('index.html', metrics=performance_metadata)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    performance_metadata = {
        "champion_model": "XGBoost Regressor Pipeline",
        "validation_rmse": 1041.50,
        "validation_mae": 714.22,
        "dataset_outlets": 10,
        "dataset_items": 1500,
        "developer_name": "Laiba Noor"
    }
    return render_template('dashboard.html', metrics=performance_metadata)

@app.route('/result', methods=['POST'])
def result():
    # 1. Capture data values from your HTML front-end interface input form
    weight = float(request.form['weight'])
    mrp = float(request.form['mrp'])
    visibility = float(request.form['visibility'])
    year = int(request.form['year'])
    
    # Apply Feature Engineering on the fly matching Chapter 10 code logic
    # Calculate operational age based on your dissertation's 2026 anchor timeline
    outlet_age = 2026 - year
    
    # Handle structural zero visibility anomalies instantly
    if visibility == 0.0:
        visibility = 0.0541 # Dataset median value placeholder

    # 2. Construct matching input dictionary array corresponding directly to model features structure
    input_data = {
        'Item_Weight': [weight],
        'Item_Visibility': [visibility],
        'Item_MRP': [mrp],
        'Outlet_Age': [outlet_age]
    }

    # Convert to Pandas DataFrame structure for algorithmic calculation processing
    df_features = pd.DataFrame(input_data)

    # 3. Calculate dynamic future sales prediction using your serialized model
    if model:
        prediction = model.predict(df_features)[0]
        final_sales = max(0, round(float(prediction), 2))
    else:
        final_sales = 0.00 # Fallback safety trace

    return render_template('result.html', prediction=final_sales)

if __name__ == '__main__':
    app.run(debug=True)
