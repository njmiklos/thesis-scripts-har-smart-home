import pandas as pd
import numpy as np
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv, get_all_csv_files_in_directory
from data_processing.convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime_srs


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

def categorize_time_of_day(hour: int) -> int:
    """
    Identifies the time of day based on hour (0-23).

    Categories:
        - 0 → morning   (05:00-10:59)
        - 1 → afternoon (11:00-15:59)
        - 2 → evening   (16:00-21:59)
        - 3 → night     (22:00-04:59)

    Args:
        hour (int): Hour of the day (0-23).

    Returns:
        int: Category label representing time of day.
    """
    if 5 <= hour < 11:
        return 0  # morning
    elif 11 <= hour < 16:
        return 1  # afternoon
    elif 16 <= hour < 22:
        return 2  # evening
    if 22 <= hour or hour < 5:
        return 3  # night
    else:
        raise ValueError(f'Cannot determine time of day of the given hour: {hour}.')
        
def add_time_of_day_as_integer(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Adds an integer-coded 'time_of_day' column to the DataFrame, based on hour extracted
    from a timestamp column in milliseconds.

    Categories:
        0 = morning (05-10)
        1 = afternoon (11-15)
        2 = evening (16-21)
        3 = night (22-04)

    Args:
        df (pd.DataFrame): DataFrame containing a timestamp column.
        time_col (str): Name of the timestamp column.

    Returns:
        pd.DataFrame: DataFrame with an added 'time_of_day' categorical column.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])

    if 'hour' in df.columns:    # Allows multiple temporal features to coexist
        df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
        df.drop(columns=['timestamp'], inplace=True)
    else:
        df['hour'] = df['timestamp'].dt.hour
        df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
        df.drop(columns=['timestamp', 'hour'], inplace=True)

    return df

def add_time_of_day_as_cyclical(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Adds sine and cosine transformations of the time of day extracted from a timestamp column.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with 'time_of_day_sin' and 'time_of_day_cos' columns added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])

    if 'time_of_day' in df.columns:    # Allows multiple temporal features to coexist
        df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 4)
        df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 4)
        df.drop(columns=['timestamp'], inplace=True)
    else:
        df = add_time_of_day_as_integer(df, time_col)
        df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 4)
        df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 4)
        df.drop(columns=['timestamp', 'time_of_day'], inplace=True)

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

def add_weekday_as_cyclical(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Adds sine and cosine transformations of the weekday extracted from a timestamp column.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with 'weekday_sin' and 'weekday_cos' columns added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])

    if 'weekday' in df.columns:    # Allows multiple temporal features to coexist
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df.drop(columns=['timestamp'], inplace=True)
    else:
        df['weekday'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df.drop(columns=['timestamp', 'weekday'], inplace=True)

    return df

def add_hour_as_cyclical(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Adds sine and cosine transformations of the hour extracted from a timestamp column.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp.
        time_col (str): Column name containing timestamps in milliseconds.

    Returns:
        pd.DataFrame: Modified DataFrame with 'hour_sin' and 'hour_cos' columns added.
    """
    df = duplicate_column(df, time_col, 'timestamp')
    df['timestamp'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['timestamp'])

    if 'hour' in df.columns:    # Allows multiple temporal features to coexist
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df.drop(columns=['timestamp'], inplace=True)
    else:
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df.drop(columns=['timestamp', 'hour'], inplace=True)

    return df

def process_files(input_dir: Path, output_dir: Path, transforms: list) -> None:
    """
    Processes all CSV files in the input directory by applying the given transform functions
    and saving the modified files to the output directory.

    Args:
        input_dir (Path): Directory containing input CSV files.
        output_dir (Path): Directory where processed CSV files will be saved.
        transforms (list): List of functions to apply to each DataFrame.
    """
    files = get_all_csv_files_in_directory(input_dir)
    for file in files:
        df = read_csv_to_dataframe(file)

        for transform in transforms:
            df = transform(df, 'time')

        filename = file.stem + '.csv'
        save_dataframe_to_csv(df, output_dir / filename)


if __name__ == '__main__':
    input_path = get_path_from_env('INPUTS_PATH')
    output_path = get_path_from_env('OUTPUTS_PATH')

    input_path_training = input_path
    input_path_testing = input_path / 'testing'

    output_path_training = output_path
    output_path_testing = output_path / 'testing'

    transforms = [
        add_day_and_month_as_integers,
        add_weekday_as_integer,
        add_weekday_as_cyclical,
        add_time_of_day_as_integer,
        add_time_of_day_as_cyclical,
        add_hour_minute_and_second_as_integers,
        add_hour_as_cyclical
    ]

    process_files(input_path_training, output_path_training, transforms)
    process_files(input_path_testing, output_path_testing, transforms)