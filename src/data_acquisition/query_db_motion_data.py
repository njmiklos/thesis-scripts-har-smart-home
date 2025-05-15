"""
Queries a running InfluxDB instance for motion sensor data (accelerometer, gyrometer, magnetometer) from a specific device 
and saves the combined results to a CSV file.

This script is tailored for use with the sensor kit deployed in the TU Chemnitz project.
"""
import pandas as pd

from utils.get_env import get_path_from_env
from utils.file_handler import save_dataframe_to_csv, check_if_directory_exists
from data_acquisition.query_db import get_query_result

if __name__ == '__main__':
    device_no = '3'
    output_filename = f'd{device_no}_motion.csv'

    outputs_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_directory_exists(outputs_dir)
    path_output_file = outputs_dir / output_filename
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

    save_dataframe_to_csv(df, path_output_file)