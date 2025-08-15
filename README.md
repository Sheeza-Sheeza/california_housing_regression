# Diabetes Progression — Linear vs Ridge vs Lasso (End-to-End)

A complete, production-style ML project using the **Diabetes Progression** dataset to compare **Linear Regression**, **Ridge**, and **Lasso** with rigorous preprocessing, feature engineering, cross-validation, diagnostics, and a minimal **Flask** API for inference.

## Highlights
- Clean project structure (`src/` with clear modules).
- Feature engineering + scaling pipelines.
- Cross-validated model selection (Linear, RidgeCV, LassoCV).
- Full metrics: MSE, RMSE, MAE, R², Adjusted R².
- Residual diagnostics, learning curve, and regularization paths.
- Saved artifacts (`models/`) + CSV reports.
- Minimal Flask API (`app_flask.py`) to serve predictions.

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train.py
```
This will:
- Download/load the Diabetes Progression dataset from `sklearn`
- Train Linear, Ridge, and Lasso with CV
- Save plots in `reports/`
- Save metrics in `reports/metrics.csv`
- Save best model to `models/best_model.joblib` and preprocessing to `models/preprocess.joblib`

## Evaluate on Test
```bash
python src/evaluate.py
```

## Run the API
```bash
python app_flask.py
```
Then POST JSON to `http://127.0.0.1:5000/predict` with keys:
```json
{
  "MedInc": 4.0,
  "HouseAge": 20.0,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 1000.0,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```

### Files
- `src/data_load.py` — load dataset as DataFrame
- `src/features.py` — feature engineering and preprocessing pipeline
- `src/train.py` — training + diagnostics + saving
- `src/evaluate.py` — test set evaluation
- `src/predict.py` — CLI prediction helper
- `app_flask.py` — minimal Flask API

> Plots are made with **matplotlib** only (no seaborn), each in its own figure, and default colors.

## License
MIT


## Data Card
**Dataset:** scikit-learn Diabetes (regression)

- **Instances:** 442
- **Features:** 10 numeric, standardized (age, sex, bmi, bp, s1–s6)
- **Target:** Disease progression one year after baseline


**Preprocessing:** Feature engineering (squares + selected interactions), StandardScaler.


**Intended Use:** Educational comparison of linear baselines (Linear/Ridge/Lasso), model selection and diagnostics.


**Ethical Notes:** This model is **not** a medical device. Predictions should not be used for clinical decisions.

