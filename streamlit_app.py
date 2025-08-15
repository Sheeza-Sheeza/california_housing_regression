import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

st.set_page_config(page_title="Diabetes Regression — Linear/Ridge/Lasso", layout="centered")

st.title("🩺 Diabetes Progression Predictor")
st.write("Enter feature values (standardized features expected by the saved pipeline). The best model from training is loaded and used for prediction.")

FEATURES = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']

# Load model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib')
model = load(MODEL_PATH)

cols = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    with cols[i % 2]:
        inputs[feat] = st.number_input(feat, value=0.0, step=0.01, format="%.4f")

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=FEATURES)
    y_pred = model.predict(X)[0]
    st.subheader("Prediction")
    st.write(f"Predicted disease progression (target units): **{y_pred:.3f}**")
    st.caption("Note: These features are normalized in the dataset; pipeline handles feature engineering + scaling.")
