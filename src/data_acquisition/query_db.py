"""
This script provides utilities to query sensor data from a running InfluxDB instance and save it to a CSV file.

It is designed for use in projects involving a specific sensor kit at TU Chemnitz. The script includes functions 
to formulate queries, retrieve data, convert timestamps, and persist the results. When run directly, it executes 
a query for a specified device and measurement and writes the results to a CSV file.
"""
import pandas as pd
from influxdb import InfluxDBClient
from typing import Any

from data_processing.convert_timestamps import convert_timestamps_from_iso8601_to_localized_datetime
from utils.get_env import get_path_from_env, get_database_info
from utils.file_handler import save_dataframe_to_csv, check_if_directory_exists


def query_data(query: str) -> Any:
    """
    Executes a query on the InfluxDB client and retrieves the result.

    Args:
        query (str): The query string to execute on the InfluxDB database.

    Returns:
        Any: The result of the query execution, typically in the form of an InfluxDB query result object.
    """
    db_host, db_port, db_name = get_database_info()
    client = InfluxDBClient(host=db_host, port=db_port)
    client.switch_database(db_name)

    query_result = client.query(query)
    return query_result

def convert_query_result_to_dataframe(query_result: Any) -> pd.DataFrame:
    """
    Converts an InfluxDB query result into a pandas DataFrame.

    Args:
        query_result (Any): The result of an InfluxDB query, containing points to be converted.

    Returns:
        pd.DataFrame: A DataFrame representation of the query result.
    """
    df = pd.DataFrame(list(query_result.get_points()))
    return df

def get_query_result(query: str) -> pd.DataFrame:
    """
    Retrieves and processes query results from InfluxDB, converting them into a pandas DataFrame
    with the 'time' column adjusted to milliseconds since the Unix epoch.

    Args:
        query (str): The query string to execute on the InfluxDB database.

    Returns:
        pd.DataFrame: A processed DataFrame with query results, including the adjusted 'time' column.
    """
    query_result = query_data(query)
    df = convert_query_result_to_dataframe(query_result)
    if not df.empty:
        df = convert_timestamps_from_iso8601_to_localized_datetime(df, 'time')
    return df

def formulate_query(device_no: str, measurement: str, values_col: str, ts_start: str = '1733353200000ms', ts_end: str = '1734476399000ms') -> str:
    """
    Generates an InfluxDB 1.x query string to retrieve sensor data.

    This function constructs a query to fetch time-series data from a typical sensor kit database 
    used in university projects. The query is designed to retrieve values from a specific sensor type 
    within a defined time range.

    Args:
        device_no (str): 
            The identifier of the sensor unit device, typically associated with a specific location.
        measurement (str): 
            The type of measurement to query, such as 'accelerometer' or other sensor types.
        values_col (str): 
            The name of the column storing the measurement values (e.g., 'x', 'y', or 'z' for an accelerometer).
        ts_start (str, optional): 
            The start timestamp for the data retrieval period. Defaults to '1733353200000ms', a recording time start for my project.
        ts_end (str, optional): 
            The end timestamp for the data retrieval period. Defaults to '1734476399000ms', a recording time end for my project.

    Returns:
        str: A formatted InfluxDB query string.

    Example:
        >>> formulate_query("1", "accelerometer", "x")
        'SELECT "time", "x" FROM "accelerometer" WHERE "deviceNo" = '1' AND time >= 1733353200000ms AND time <= 1734476399000ms;'
    """
    query = f"""
    SELECT "time", "{values_col}" FROM "{measurement}" WHERE "deviceNo" = '{device_no}' AND time >= {ts_start} AND time <= {ts_end};
    """
    return query


if __name__ == '__main__':
    outputs_dir = get_path_from_env('OUTPUTS_PATH')

    # Set before running
    device_no = '1'
    measurement = 'accelerometer'
    values_col = 'x'

    check_if_directory_exists(outputs_dir)
    output_filename = f'd{device_no}_{measurement}.csv'
    output_file_path = outputs_dir / output_filename
    query = formulate_query(device_no, measurement, values_col)
    df = get_query_result(query)

    save_dataframe_to_csv(df, output_file_path)