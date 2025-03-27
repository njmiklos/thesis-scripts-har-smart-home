import pandas as pd
from pathlib import Path

from get_env import (get_input_path, get_output_path)
from handle_csv import (read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory)
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime_srs


def duplicate_column(df: pd.DataFrame, time_col: str, name: str) -> pd.DataFrame:
    """
    Creates a copy of the specified time column under a new column name.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the original column.
        time_col (str): Name of the existing column to copy.
        name (str): Name of the new column.
    
    Returns:
        pd.DataFrame: Modified DataFrame with the new duplicated column.
    """
    df[name] = df[time_col].copy()
    return df

def add_day_and_month_as_integers(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Extracts day and month integers from a timestamp column and adds them as new columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with 'day' and 'month' integer columns added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])  # 2000-01-01 01:00:00+01:00

    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    df.drop(columns=['timestamp'], inplace=True)
    return df

def add_weekday_as_integer(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Extracts the weekday (0=Monday, 6=Sunday) from a timestamp column 
    and adds it as a new column called 'weekday'.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with the 'weekday' integer column added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])  # 2000-01-01 01:00:00+01:00

    df['weekday'] = df['timestamp'].dt.dayofweek

    df.drop(columns=['timestamp'], inplace=True)
    return df

def add_hour_minute_and_second_as_integers(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Extracts hour, minute, and second integers from a timestamp column and adds them as new columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with 'hour', 'minute', and 'second' integer columns added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])  # 2000-01-01 01:00:00+01:00

    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second

    df.drop(columns=['timestamp'], inplace=True)
    return df

def process_files(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all CSV files in the input directory by extracting date and time components 
    and saving the modified files to the output directory.

    Args:
        input_dir (Path): Path to the directory containing input CSV files.
        output_dir (Path): Path to the directory where processed CSV files will be saved.
    """
    files = get_all_csv_files_in_directory(input_dir)
    for file in files:
        df = read_csv_to_pandas_dataframe(file)

        df = add_day_and_month_as_integers(df, 'time')
        df = add_hour_minute_and_second_as_integers(df, 'time')

        filename = file.stem
        save_pandas_dataframe_to_csv(df, output_dir / filename)


if __name__ == '__main__':
    input_path = get_input_path()
    output_path = get_output_path()

    input_path_training = input_path
    input_path_testing = input_path / 'testing'

    output_path_training = output_path
    output_path_testing = output_path / 'testing'

    process_files(input_path_training, output_path_training)
    process_files(input_path_testing, output_path_testing)