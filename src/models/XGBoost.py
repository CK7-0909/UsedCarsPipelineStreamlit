from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import joblib as jb


def train_model(df: pd.DataFrame) -> Pipeline:
    df = df.copy()
    # df.columns = [col.lower() for col in df.columns]

    # expected_columns = ["manufacturer", "PRICE", "YEAR", "MODEL", "ODOMETER", "TITLE_STATUS", "TRANSMISSION", "PAINT_COLOR", "STATE"]
    # Ecoding categorical variables 
    categorical_cols = ["manufacturer", "model", "title_status", "transmission", "paint_color", "state"]
    X = pd.get_dummies(df[["age", "odometer"] + categorical_cols], drop_first=True) # get_dummies to convert categorical variables to numerical
    y = np.log(df["price"]) # log transformation to reduce the impact of outliers. It helps the model learn percentage-based relationships (e.g., “each year reduces value by ~10%”) rather than absolute drops (“each year reduces $1,500”). 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model configuration optimized by GPT, review again later
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
        )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_true = np.exp(y_test)
    y_pred = np.exp(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred) # average absolute difference between predicted and actual values
    rmse = root_mean_squared_error(y_true, y_pred) # square root of the average squared differences between predicted and actual values (more sensitive to large errors)
    r2 = r2_score(y_true, y_pred) # try to main around .8 or higher

    # print(f"MAE: {mae:.1f}  RMSE: {rmse:.1f}  R^2: {r2:.3f}")

    # errors = abs(y_true - y_pred) / y_true * 100
    # print("Mean % Error:", errors.mean())
    # print("Median % Error:", errors.median())

    # Save the trained model to a file
    jb.dump(model, "models/xgb_model_all.joblib")

    # Save predictions to reduce rerunning model training
    eval_df = pd.DataFrame({
    "actual_price": y_true.values,
    "predicted_price": y_pred,
    "abs_error": abs(y_true - y_pred) / y_true * 100
    })
    eval_df.to_parquet("models/evaluation_all.parquet", index=False)

    return y_pred, y_true
