# Take inputs from streamlit and run the model to preict the price
import joblib
import numpy as np
import pandas as pd
from src.data_processing.preprocessing import feature_engineer

Model_Path = "models/xgb_model_all.joblib"

def inputDataPrediction(prediction_df: pd.DataFrame):

    df = feature_engineer(prediction_df)
    model = joblib.load(Model_Path)

    # Ecoding categorical variables 
    categorical_cols = ["manufacturer", "model", "title_status", "transmission", "paint_color", "state"]
    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.astype(str).str.strip().str.lower())
    feature_frame = pd.get_dummies(df[["age", "odometer"] + categorical_cols], drop_first=False) # get_dummies to convert categorical variables to numerical
 
    # The model expects the exact dummy-coded feature array from training, so we need to ensure all columns are present
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = model.get_booster().feature_names or []
    feature_names = list(feature_names)
    feature_frame = feature_frame.reindex(columns=feature_names, fill_value=0)
  
    predictions = model.predict(feature_frame)
    return np.exp(predictions) # reverse log transformation
