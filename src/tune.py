import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from joblib import dump
from src.data_load import load_california_housing
from src.features import make_preprocess_pipeline

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

def main():
    df, feature_names, target_col = load_california_housing(as_frame=True)
    X = df[feature_names].copy()
    y = df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = make_preprocess_pipeline(feature_names)

    ridge = Pipeline([('prep', preprocess), ('model', Ridge(random_state=RANDOM_STATE))])
    lasso = Pipeline([('prep', preprocess), ('model', Lasso(random_state=RANDOM_STATE, max_iter=10000))])

    param_grid_ridge = {'model__alpha': np.logspace(-3, 3, 30)}
    param_grid_lasso = {'model__alpha': np.logspace(-3, 1, 30)}

    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    gs_ridge = GridSearchCV(ridge, param_grid_ridge, cv=cv, scoring='r2', n_jobs=None, return_train_score=True)
    gs_lasso = GridSearchCV(lasso, param_grid_lasso, cv=cv, scoring='r2', n_jobs=None, return_train_score=True)

    gs_ridge.fit(X_train, y_train)
    gs_lasso.fit(X_train, y_train)

    # Save CV results
    ridge_df = pd.DataFrame(gs_ridge.cv_results_)
    lasso_df = pd.DataFrame(gs_lasso.cv_results_)
    ridge_df.to_csv(os.path.join(REPORTS_DIR, 'ridge_gridsearch_results.csv'), index=False)
    lasso_df.to_csv(os.path.join(REPORTS_DIR, 'lasso_gridsearch_results.csv'), index=False)

    # Plot validation curves (alpha vs mean_test_score)
    def plot_curve(df, title, out):
        alphas = df['param_model__alpha'].astype(float)
        scores = df['mean_test_score']
        plt.figure()
        plt.semilogx(alphas, scores, marker='o')
        plt.xlabel('alpha (log scale)')
        plt.ylabel('Mean CV R²')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    plot_curve(ridge_df, 'Ridge Validation Curve (alpha vs CV R²)', os.path.join(REPORTS_DIR, 'ridge_validation_curve.png'))
    plot_curve(lasso_df, 'Lasso Validation Curve (alpha vs CV R²)', os.path.join(REPORTS_DIR, 'lasso_validation_curve.png'))

    # Save best estimators (pipelines)
    dump(gs_ridge.best_estimator_, os.path.join(MODELS_DIR, 'ridge_best_grid.joblib'))
    dump(gs_lasso.best_estimator_, os.path.join(MODELS_DIR, 'lasso_best_grid.joblib'))

    # Write a summary
    with open(os.path.join(REPORTS_DIR, 'gridsearch_summary.txt'), 'w') as f:
        f.write(f"Best Ridge alpha: {gs_ridge.best_params_['model__alpha']}, CV R²: {gs_ridge.best_score_}\n")
        f.write(f"Best Lasso alpha: {gs_lasso.best_params_['model__alpha']}, CV R²: {gs_lasso.best_score_}\n")

if __name__ == '__main__':
    main()
