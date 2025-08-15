import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_validate, KFold, learning_curve
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from joblib import dump

from src.data_load import load_california_housing
from src.features import make_preprocess_pipeline

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val, cv=5):
    scoring = ('neg_mean_squared_error','neg_mean_absolute_error','r2')
    cvres = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False)
    mse_cv = -cvres['test_neg_mean_squared_error'].mean()
    mae_cv = -cvres['test_neg_mean_absolute_error'].mean()
    r2_cv = cvres['test_r2'].mean()
    rmse_cv = math.sqrt(mse_cv)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    mse_v = mean_squared_error(y_val, preds)
    mae_v = mean_absolute_error(y_val, preds)
    rmse_v = math.sqrt(mse_v)
    r2_v = r2_score(y_val, preds)
    adj_r2_v = adjusted_r2(r2_v, n=len(y_val), k=pipeline[-1].coef_.shape[0] if hasattr(pipeline[-1], 'coef_') else X_train.shape[1])

    metrics = {
        'model': name,
        'cv_mse': mse_cv,
        'cv_rmse': rmse_cv,
        'cv_mae': mae_cv,
        'cv_r2': r2_cv,
        'val_mse': mse_v,
        'val_rmse': rmse_v,
        'val_mae': mae_v,
        'val_r2': r2_v,
        'val_adj_r2': adj_r2_v,
    }
    return pipeline, preds, metrics

def plot_residuals(y_true, y_pred, title, path):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, s=8)
    plt.axhline(0)
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

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

def plot_learning_curve(estimator, X, y, title, path):
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='r2', n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Training R²')
    plt.plot(train_sizes, test_scores.mean(axis=1), marker='s', label='CV R²')
    plt.xlabel('Training examples')
    plt.ylabel('R²')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_regularization_paths(preprocess, X, y):
    # Ridge path
    alphas = np.logspace(-3, 3, 30)
    coefs_ridge = []
    Xp = preprocess.fit_transform(X)
    for a in alphas:
        ridge = RidgeCV(alphas=[a], cv=5)
        ridge.fit(Xp, y)
        coefs_ridge.append(ridge.coef_)
    plt.figure()
    for i in range(np.array(coefs_ridge).shape[1]):
        plt.plot(alphas, np.array(coefs_ridge)[:, i])
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title('Ridge Regularization Path (per-feature)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'ridge_regularization_path.png'))
    plt.close()

    # Lasso path
    alphas_lasso = np.logspace(-3, 1, 30)
    coefs_lasso = []
    for a in alphas_lasso:
        lcv = LassoCV(alphas=[a], cv=5, random_state=RANDOM_STATE, max_iter=10000)
        lcv.fit(Xp, y)
        coefs_lasso.append(lcv.coef_)
    plt.figure()
    for i in range(np.array(coefs_lasso).shape[1]):
        plt.plot(alphas_lasso, np.array(coefs_lasso)[:, i])
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title('Lasso Regularization Path (per-feature)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'lasso_regularization_path.png'))
    plt.close()

def main():
    df, feature_names, target_col = load_california_housing(as_frame=True)
    df[target_col] = np.log1p(df[target_col])  # log-transform target

    X = df[feature_names].copy()
    y = df[target_col].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = make_preprocess_pipeline(feature_names)

    # Models
    linear = Pipeline([('prep', preprocess), ('model', LinearRegression())])
    ridge = Pipeline([('prep', preprocess), ('model', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
    lasso = Pipeline([('prep', preprocess), ('model', LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000))])

    results = []
    fitted = {}

    for name, pipe in [('Linear', linear), ('RidgeCV', ridge), ('LassoCV', lasso)]:
        model, preds, metrics = evaluate_model(name, pipe, X_train, y_train, X_val, y_val, cv=5)
        results.append(metrics)
        fitted[name] = (model, preds)
        plot_residuals(y_val, preds, f'{name} Residuals', os.path.join(REPORTS_DIR, f'{name.lower()}_residuals.png'))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_DIR, 'metrics.csv'), index=False)

    best_name = results_df.sort_values('val_r2', ascending=False).iloc[0]['model']
    best_model = fitted[best_name][0]

    plot_learning_curve(best_model, X_train, y_train, f'{best_name} Learning Curve', os.path.join(REPORTS_DIR, 'learning_curve.png'))
    plot_pred_vs_actual(y_val, fitted[best_name][1], f'{best_name} Predicted vs Actual', os.path.join(REPORTS_DIR, 'pred_vs_actual.png'))

    resid_best = y_val - fitted[best_name][1]
    plot_residual_hist(resid_best, f'{best_name} Residual Distribution', os.path.join(REPORTS_DIR, 'residual_hist.png'))

    # Regularization paths
    prep_only = clone(preprocess)
    plot_regularization_paths(prep_only, X_train, y_train)

    # Save artifacts
    dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
    dump(preprocess, os.path.join(MODELS_DIR, 'preprocess.joblib'))

    with open(os.path.join(REPORTS_DIR, 'best_model.txt'), 'w') as f:
        f.write(str(best_name))

    print('Training complete. Results:\n', results_df)

if __name__ == "__main__":
    main()
