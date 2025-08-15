from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Directories
PROJECT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Feature order must match training
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude']

# Load the trained pipeline once at startup
model = load(os.path.join(MODELS_DIR, 'best_model.joblib'))

@app.route('/', methods=['GET'])
def home():
    return "California Housing Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):  # Single record
            data = [data]
        df = pd.DataFrame(data, columns=FEATURES)

        # Model handles preprocessing internally
        y_log1p = model.predict(df)

        # Invert log1p transformation
        y = np.expm1(y_log1p)

        return jsonify({"predictions": y.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
