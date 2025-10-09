import pandas as pd
from src.data_processing.preprocessing import testing_feature_engineer

df = pd.read_json("data/sample_data/compass.json")
print('raw columns:', df.columns.tolist())

df = testing_feature_engineer(df)
print('post feature:', df.columns.tolist())

categorical_cols = ["title_status", "transmission", "paint_color", "state"]
X = pd.get_dummies(df[["age", "odometer"] + categorical_cols], drop_first=True)
print('dummy columns:', X.columns.tolist())
