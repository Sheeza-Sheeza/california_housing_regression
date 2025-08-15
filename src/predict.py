import json, sys, os
from joblib import load
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

FEATURES = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py '{"MedInc":4,"HouseAge":20,"AveRooms":5,"AveBedrms":1,"Population":1000,"AveOccup":3,"Latitude":34,"Longitude":-118}'")
        sys.exit(1)
    payload = json.loads(sys.argv[1])
    x = pd.DataFrame([payload], columns=FEATURES)
    model = load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    y_log1p = model.predict(x)[0]
    # invert log1p
    y_pred = np.expm1(y_log1p)
    print({"predicted_median_house_value": float(y_pred)})

if __name__ == "__main__":
    main()
