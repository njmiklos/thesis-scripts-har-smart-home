import pandas as pd

from get_env import get_base_path
from handle_csv import (save_pandas_dataframe_to_csv)
from query_db import get_query_result

if __name__ == '__main__':
    base_path = get_base_path()
    device_no = '3'
    path_output_file = base_path / f'd{device_no}_motion.csv'

    queries = {
        "accelerometer" : f""" 
        SELECT "time", "x" AS "accX", "y" AS "accY", "z" AS "accZ" FROM "accelerometer" WHERE "deviceNo" = '{device_no}' AND time >= 1733353200000ms AND time <= 1734476399000ms;
        """,
        "gyrometer" : f"""
        SELECT "time", "x" AS "gyroX", "y" AS "gyroY", "z" AS "gyroZ" FROM "gyrometer" WHERE "deviceNo" = '{device_no}' AND time >= 1733353200000ms AND time <= 1734476399000ms;
        """,
        "magnetometer" : f"""
        SELECT "time", "x" AS "magX", "y" AS "magY", "z" AS "magZ" FROM "magnetometer" WHERE "deviceNo" = '{device_no}' AND time >= 1733353200000ms AND time <= 1734476399000ms;
        """
    }

    df = None
    for label, query in queries.items():
        query_result = get_query_result(query)
        if df is None:
            df = query_result
        else:
            df = pd.merge(df, query_result, on="time", how="outer") # outer includes all rows. If time does not match, filled with NaN

    save_pandas_dataframe_to_csv(df, path_output_file)