import os, math
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_load import load_california_housing

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def main():
    df, feature_names, target_col = load_california_housing(as_frame=True)
    df[target_col] = np.log1p(df[target_col])
    # hold out last 20% as test to mimic unseen data (same split as train by random_state)
    from sklearn.model_selection import train_test_split
    X = df[feature_names].copy()
    y = df[target_col].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    preds = best_model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    k = best_model[-1].coef_.shape[0] if hasattr(best_model[-1], 'coef_') else X_test.shape[1]
    adj_r2 = adjusted_r2(r2, n=len(y_test), k=k)

    metrics = pd.DataFrame([{
        'set': 'test',
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2,
    }])
    metrics.to_csv(os.path.join(REPORTS_DIR, 'test_metrics.csv'), index=False)
    print(metrics)

if __name__ == "__main__":
    main()
