import pandas as pd
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import get_all_csv_files_in_directory, read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from data_processing.convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime, convert_timestamps_from_localized_datetime_to_miliseconds

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

def generate_new_timestamp_index(df: pd.DataFrame, frequency: str, global_start: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Generate a new, uniformly spaced timestamp index for synchronization.

    The function creates a new datetime index starting from the provided global start time,
    ending at the maximum timestamp present in the DataFrame's index, and using a fixed frequency.
    The resulting index is localized to the 'Europe/Berlin' timezone.

    Args:
        df (pd.DataFrame): The DataFrame whose index contains datetime information.
        frequency (str): A frequency string (e.g., '1S', '100ms') compatible with pandas' date_range.
        global_start (pd.Timestamp): The start time for the new index, typically the earliest timestamp
                                     across all files.

    Returns:
        pd.DatetimeIndex: A new datetime index spanning from global_start to the maximum timestamp in df.
    """
    ts_end = df.index.max()
    new_index = pd.date_range(start=global_start, end=ts_end, freq=frequency, tz='Europe/Berlin')
    return new_index

def resample_and_align_data(df: pd.DataFrame, new_index: pd.DatetimeIndex, frequency: str, magnitude_data: bool = False) -> pd.DataFrame:
    """
    Resample the DataFrame to a uniform interval and align it with a new timestamp index.

    The function first resamples the DataFrame using the specified frequency by computing the average
    of aggregated samples. It then reindexes the resampled DataFrame to the provided new_index.
    Gaps in the data are filled either with bfill and ffill for most data, or with 0s for magnitude data, 
    ensuring continuous data coverage.

    Args:
        df (pd.DataFrame): The original DataFrame indexed by time.
        new_index (pd.DatetimeIndex): The target datetime index for alignment.
        frequency (str): The resampling frequency (e.g., '1s', '100ms').
        magnitude_data (bool): A flag to set for different handling of missing values for 0-padded magnitude data. 

    Returns:
        pd.DataFrame: The DataFrame resampled and aligned to the new timestamp index.
    """
    df_new = df.resample(frequency).mean()  # Resample and aggregate
    df_new = df_new.reindex(new_index).interpolate(method='time')   # Align data to new_index

    # Fill missing values after reindexing
    if magnitude_data:
        df_new = df_new.fillna(0)
    else:
        df_new = df_new.ffill().bfill()

    return df_new

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

def process_file(file_path: Path, output_path: Path, frequency: str, global_start: pd.Timestamp):
    """
    Process and synchronize the timestamps in a CSV file, then save the updated file.

    Args:
        file_path (Path): The path to the CSV file to process.
        output_path (Path): The directory where the synchronized CSV file will be saved.
        frequency (str): The resampling frequency (e.g., '1S', '100ms') used to align the timestamps.
        global_start (pd.Timestamp): The starting timestamp to use for generating the new index.

    Returns:
        None
    """
    df = read_csv_to_pandas_dataframe(file_path)

    if not df.empty:
        df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')

        df = set_time_index(df, 'time')
        new_index = generate_new_timestamp_index(df, frequency, global_start)

        if 'magnitude' in str(file_path):
            df_synchronized = resample_and_align_data(df, new_index, frequency, magnitude_data=True)
        else:
            df_synchronized = resample_and_align_data(df, new_index, frequency)

        df_synchronized = restore_time_column(df_synchronized)
        df_synchronized = convert_timestamps_from_localized_datetime_to_miliseconds(df_synchronized, 'time')

        new_file_path = output_path / file_path.name
        save_pandas_dataframe_to_csv(df_synchronized, new_file_path)
        print(f'Saved synchronized file: {output_path / file_path.name}')
    
    else:
        print(f'Warning: {file_path} is empty. Skipping...')


if __name__ == '__main__':
    base_path = get_path_from_env('BASE_PATH')
    dataset_path = base_path / 'Sync_ready'
    new_dataset_path = base_path / 'Synchronized'
    frequency = '1s'

    files = get_all_csv_files_in_directory(dataset_path)

    if files:

        global_start_time = get_global_start_time(files)
        print(f'INFO: Setting global start time for all files: {global_start_time}')

        no_files = len(files)
        counter = 1
        for file in files:
            print(f'INFO: Processing file {counter}/{no_files} {file.name}...')
            process_file(file, new_dataset_path, frequency=frequency, global_start=global_start_time)
            counter = counter + 1

        print('INFO: Synchronization complete.')
    else:
        print('WARNING: No CSV files found. Exiting.')