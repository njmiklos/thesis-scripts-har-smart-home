'''
synchronize_data.py:
The function generate_new_timestamp_index creates a new timestamp index starting from the global start time 
and ending at the maximum timestamp in each individual file. Since different files may have different maximum timestamps, 
the resulting timestamps will extend to different points, leading to varying lengths.

This file is a draft at traying to mend it.
'''

import pandas as pd
from pathlib import Path

from get_env import get_base_path
from handle_csv import get_all_csv_files_in_directory, read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime, convert_timestamps_from_localized_datetime_to_miliseconds


def get_global_start_time(files: list) -> pd.Timestamp:
    """
    Determine the earliest timestamp across multiple CSV files.

    This function iterates over a list of file paths, reads each CSV file into a DataFrame,
    converts the 'time' column from milliseconds to a localized datetime (using the provided helper),
    and then finds the minimum (earliest) timestamp present in each file. The global minimum timestamp
    is returned.

    Args:
        files (List[Path]): A list of file paths to CSV files.

    Returns:
        pd.Timestamp: The earliest timestamp found across all files.
                     If none of the files contain data, the function returns None.
    """
    min_timestamp = None

    for file_path in files:
        df = read_csv_to_pandas_dataframe(file_path)
        if not df.empty:
            df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')
            file_min_time = df['time'].min()

            if min_timestamp is None or file_min_time < min_timestamp:
                min_timestamp = file_min_time

    return min_timestamp

def get_global_end_time(files: list) -> pd.Timestamp:
    """
    Determine the latest timestamp across multiple CSV files.

    Args:
        files (List[Path]): A list of file paths to CSV files.

    Returns:
        pd.Timestamp: The latest timestamp found across all files.
    """
    max_timestamp = None

    for file_path in files:
        df = read_csv_to_pandas_dataframe(file_path)
        if not df.empty:
            df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')
            file_max_time = df['time'].max()

            if max_timestamp is None or file_max_time > max_timestamp:
                max_timestamp = file_max_time

    return max_timestamp


def generate_global_timestamp_index(global_start: pd.Timestamp, global_end: pd.Timestamp, frequency: str) -> pd.DatetimeIndex:
    """
    Generate a globally consistent timestamp index.

    Args:
        global_start (pd.Timestamp): The earliest timestamp.
        global_end (pd.Timestamp): The latest timestamp.
        frequency (str): The resampling frequency (e.g., '1S', '100ms').

    Returns:
        pd.DatetimeIndex: A new datetime index spanning from global_start to global_end.
    """
    return pd.date_range(start=global_start, end=global_end, freq=frequency, tz='Europe/Berlin')


def resample_and_align_data_with_global_index(df: pd.DataFrame, global_index: pd.DatetimeIndex, magnitude_data: bool = False) -> pd.DataFrame:
    """
    Resample the DataFrame to a uniform interval and align it with the global timestamp index.

    Args:
        df (pd.DataFrame): The original DataFrame indexed by time.
        global_index (pd.DatetimeIndex): The global datetime index for alignment.
        magnitude_data (bool): Whether to fill missing values with zeros for magnitude data.

    Returns:
        pd.DataFrame: The DataFrame resampled and aligned to the global timestamp index.
    """
    df_new = df.resample(global_index.freq).mean()  # Resample and aggregate
    df_new = df_new.reindex(global_index).interpolate(method='time')   # Align data to global index

    # Fill missing values after reindexing
    if magnitude_data:
        df_new = df_new.fillna(0)
    else:
        df_new = df_new.bfill().ffill()

    return df_new

def set_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Set the specified column as the DataFrame index.

    This function sets the given column (typically a timestamp) as the index of the DataFrame.
    The operation is done in place and the modified DataFrame is returned.

    Args:
        df (pd.DataFrame): The DataFrame containing the time column.
        time_col (str): The name of the column to set as the index.

    Returns:
        pd.DataFrame: The DataFrame with the specified column set as its index.
    """
    df.set_index(time_col, inplace=True)
    return df

def restore_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset the DataFrame index to a column and rename it to 'time'.

    This function resets the current index of the DataFrame, which is assumed to contain timestamp data,
    and then renames the resulting column from 'index' to 'time' to restore it as a regular column.

    Returns:
        pd.DataFrame: The DataFrame with the former index now restored as a column named 'time'.
    """
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    return df

def process_file_with_global_index(file_path: Path, output_path: Path, global_index: pd.DatetimeIndex):
    """
    Process and synchronize the timestamps in a CSV file using the global index.

    Args:
        file_path (Path): The path to the CSV file to process.
        output_path (Path): The directory where the synchronized CSV file will be saved.
        global_index (pd.DatetimeIndex): The global timestamp index for synchronization.

    Returns:
        None
    """
    df = read_csv_to_pandas_dataframe(file_path)

    if not df.empty:
        df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')

        df = set_time_index(df, 'time')

        if 'magnitude' in str(file_path):
            df_synchronized = resample_and_align_data_with_global_index(df, global_index, magnitude_data=True)
        else:
            df_synchronized = resample_and_align_data_with_global_index(df, global_index)

        df_synchronized = restore_time_column(df_synchronized)
        df_synchronized = convert_timestamps_from_localized_datetime_to_miliseconds(df_synchronized, 'time')

        new_file_path = output_path / file_path.name
        save_pandas_dataframe_to_csv(df_synchronized, new_file_path)
        print(f'Saved synchronized file: {output_path / file_path.name}')
    
    else:
        print(f'Warning: {file_path} is empty. Skipping...')


if __name__ == '__main__':
    base_path = get_base_path()
    dataset_path = base_path / 'Sync_ready'
    new_dataset_path = base_path / 'Synchronized'
    frequency = '1s'

    files = get_all_csv_files_in_directory(dataset_path)

    if files:

        global_start_time = get_global_start_time(files)
        global_end_time = get_global_end_time(files)
        print(f'INFO: Setting global start time: {global_start_time} and global end time: {global_end_time}')

        global_index = generate_global_timestamp_index(global_start_time, global_end_time, frequency)

        no_files = len(files)
        counter = 1
        for file in files:
            print(f'INFO: Processing file {counter}/{no_files} {file.name}...')
            process_file_with_global_index(file, new_dataset_path, global_index)
            counter += 1

        print('INFO: Synchronization complete with uniform length for all files.')
    else:
        print('WARNING: No CSV files found. Exiting.')