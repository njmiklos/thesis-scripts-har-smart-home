import pandas as pd

from typing import Set, List

from get_env import get_input_path, get_output_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from convert_timestamps import (convert_timestamps_from_miliseconds_to_localized_datetime_srs, 
                            convert_timestamps_from_localized_datetime_to_miliseconds_srs)


def filter_by_date(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the rows from the specified day.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        day (str): The date to filter by in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A DataFrame with rows corresponding to the specified day.
    """
    df['time'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['time'])
    df = df[df['time'].dt.date == pd.to_datetime(day).date()]
    df['time'] = convert_timestamps_from_localized_datetime_to_miliseconds_srs(df['time'])
    return df

def filter_by_time_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only rows within the specified time range.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        start (str): The start time in the format 'HH:MM:SS+TZ'.
        end (str): The end time in the format 'HH:MM:SS+TZ'.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only rows within the specified time range.
    """
    df['time'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['time'])
    df = df[(df['time'].dt.time >= pd.to_datetime(start).time()) & 
        (df['time'].dt.time <= pd.to_datetime(end).time())]
    df['time'] = convert_timestamps_from_localized_datetime_to_miliseconds_srs(df['time'])
    return df

def filter_by_timestamp(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Returns a DataFrame containing only rows within the specified timestamp range.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        start (int): The start time in milliseconds.
        end (int): The end time in milliseconds.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only rows within the specified time range.
    """
    filtered_df = df[(df['time'] >= start) & (df['time'] <= end)]
    return filtered_df

def remove_rows_with_annotation(df: pd.DataFrame, annotation: str) -> pd.DataFrame:
    """
    Removes rows containing the specified annotation.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        annotation (str): The annotation to filter out.

    Returns:
        pd.DataFrame: A DataFrame with rows containing the specified annotation removed.
    """
    df = df[~df['annotation'].str.contains(annotation, na=False)]
    return df

def remove_rows_with_annotations(df: pd.DataFrame, annotations: Set[str]) -> pd.DataFrame:
    """
    Filters out rows containing any of the specified annotations.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        annotations (Set[str]): A set of annotations to filter out.

    Returns:
        pd.DataFrame: A DataFrame with rows containing any of the specified annotations removed.
    """
    for annotation in annotations:
        df = remove_rows_with_annotation(df, annotation)
    return df

def remove_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Removes the specified column from the DataFrame if it exists.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        col_name (str): The name of the column to remove.

    Returns:
        pd.DataFrame: A DataFrame with the specified column removed if it exists.
    """
    df = df.drop(columns=col_name, errors='ignore')
    return df

def remove_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Removes the specified columns from the DataFrame if they exist.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        columns (List[str]): A list of column names to remove.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns removed if they exist.
    """
    for col in columns:
        df = remove_column(df, col)
    return df

def select_columns(df: pd.DataFrame, relevant_columns: List[str]) -> pd.DataFrame:
    """
    Keeps only the specified columns from the DataFrame, dropping all others.
    It raises an error if any of the specified columns do not exist in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        relevant_columns (List[str]): A list of column names to keep.

    Returns:
        pd.DataFrame: A DataFrame containing only the specified columns that exist in the original DataFrame.
    """
    if relevant_columns:
        df = df[relevant_columns]
    return df