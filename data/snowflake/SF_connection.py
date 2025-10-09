import snowflake.connector as sc
import pandas as pd
import src.config.config as config

def get_dataframe(query: str) -> pd.DataFrame:
    conn = sc.connect(
        user=config.SNOWFLAKE_USER,
        password=config.SNOWFLAKE_PASSWORD,
        account=config.SNOWFLAKE_ACCOUNT,
        warehouse=config.SNOWFLAKE_WH,
        database=config.SNOWFLAKE_DB,
        schema=config.SNOWFLAKE_SCHEMA
    )
    cs = conn.cursor()
    cs.execute(query)
    rows = cs.fetchall()
    col_names = [desc[0] for desc in cs.description]
    df = pd.DataFrame(rows, columns=col_names)
    cs.close()
    conn.close()
    return df

def query_to_df():
    query = "SELECT dv.manufacturer, fv.price, dv.year, dv.model, fv.odometer, fv.title_status, dv.transmission, fv.paint_color, dl.state FROM fact_vehicles fv LEFT JOIN dim_vehicle dv on dv.vehicle_id = fv.dim_vehicle_id LEFT JOIN dim_location dl on dl.location_id = fv.dim_location_id"
    raw_df = get_dataframe(query)
    
    expected_columns = ["MANUFACTURER", "PRICE", "YEAR", "MODEL", "ODOMETER", "TITLE_STATUS", "TRANSMISSION", "PAINT_COLOR", "STATE"]
    missing = [col for col in expected_columns if col not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing expected columns from Snowflake query: {missing}")

    df = raw_df.loc[:, expected_columns].rename(columns=str.lower)
    return df
