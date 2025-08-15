import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Generic engineered features that adapt to available columns
    for col in X.columns:
        X[f'{col}_sq'] = X[col] ** 2
    # Add a few meaningful interactions if present
    def add_interaction(a, b, name):
        if a in X.columns and b in X.columns:
            X[name] = X[a] * X[b]
    add_interaction('bmi', 'bp', 'bmi_x_bp')
    add_interaction('bmi', 's5', 'bmi_x_s5')
    add_interaction('bp', 's5', 'bp_x_s5')
    return X

def make_preprocess_pipeline(feature_names):
    fe = FunctionTransformer(_engineer_features, validate=False, feature_names_out="one-to-one")
    preprocess = Pipeline(steps=[
        ('fe', fe),
        ('scale', StandardScaler(with_mean=True, with_std=True))
    ])
    return preprocess
