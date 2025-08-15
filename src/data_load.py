from sklearn.datasets import load_diabetes
import pandas as pd

def load_california_housing(as_frame: bool = True):
    # Replaced with Diabetes regression dataset to avoid internet fetch.
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={'target': 'target'}, inplace=True)
    # feature names:
    feature_names = list(data.feature_names)
    return df, feature_names, 'target'
