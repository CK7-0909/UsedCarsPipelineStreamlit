import joblib
import numpy as np
import pandas as pd
from src.data_processing.preprocessing import feature_engineer

Model_Path = "models/xgb_model_all.joblib"

Sample_data_path = "data/sample_data/compass.json"

def modelTesting(test_df: pd.DataFrame):
    df = feature_engineer(test_df)
    categorical_cols = ["manufacturer", "model", "title_status", "transmission", "paint_color", "state"]
    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.astype(str).str.strip().str.lower())
    model = joblib.load(Model_Path)

    # Ecoding categorical variables 
    feature_frame = pd.get_dummies(df[["age", "odometer"] + categorical_cols], drop_first=False) # get_dummies to convert categorical variables to numerical
 
    # The model expects the exact dummy-coded feature array from training, so we need to ensure all columns are present
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = model.get_booster().feature_names or []

    for col in feature_names:
        if col not in feature_frame.columns:
            feature_frame[col] = 0

    feature_frame = feature_frame.reindex(columns=feature_names, fill_value=0)
  
    predictions = model.predict(feature_frame)

    return np.exp(predictions) # reverse log transformation


def sample_test():
    test_df = pd.read_json(Sample_data_path)
    predictions = modelTesting(test_df)
    print("Predicted Prices:", predictions)

