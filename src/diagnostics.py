import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
from src.data_load import load_california_housing

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42

def plot_pred_vs_actual(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    minv = min(y_true.min(), y_pred.min())
    maxv = max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_residual_hist(residuals, title, path):
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    df, feature_names, target_col = load_california_housing(as_frame=True)
    X = df[feature_names].copy()
    y = df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    y_pred = model.predict(X_val)

    plot_pred_vs_actual(y_val, y_pred, 'Best Model Predicted vs Actual', os.path.join(REPORTS_DIR, 'pred_vs_actual.png'))
    residuals = y_val - y_pred
    plot_residual_hist(residuals, 'Best Model Residual Distribution', os.path.join(REPORTS_DIR, 'residual_hist.png'))

if __name__ == '__main__':
    main()
