from data.snowflake.SF_connection import query_to_df
from src.data_processing.preprocessing import model_feature_engineer
from src.models.XGBoost import train_model
from src.streamlit.ploty import plot_actual_vs_predicted, inputData
from tests.modelTesting import modelTesting, sample_test
import streamlit as st
from src.predictions.inputDataPrediction import inputDataPrediction
import pandas as pd

def main():
    # Load data from Snowflake
    df = query_to_df()

    # Preprocess data
    df = model_feature_engineer(df)
    # print(len(df), "records after feature engineering")

    # df = df.corr(numeric_only=True) # Correlation matrix to identify most useful features

    # Train model
    y_pred, y_true = train_model(df)

    # Test model
    # sample_test()

    # Visualize results
    # eval_df = pd.read_parquet("models/evaluation_all.parquet")
    # plot_actual_vs_predicted(eval_df["actual_price"], eval_df["predicted_price"])
    plot_actual_vs_predicted(y_true, y_pred) # Use this line if you want to see results from current training session

    input_data = inputData()
    st.write("Predicted Price", inputDataPrediction(input_data))
   
if __name__ == "__main__":
    main()
