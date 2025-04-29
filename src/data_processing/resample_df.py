import pandas as pd
import time

from pathlib import Path

from data_processing.convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime, convert_timestamps_from_localized_datetime_to_miliseconds
from utils.handle_csv import (read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory)
from utils.get_env import get_base_path
from data_processing.infer_sensor_metadata import infer_precision, infer_expected_sampling_rate


def interpolate_missing_values(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    #df_resampled = df.resample(interval).asfreq()
    df_resampled = df.resample(interval).asfreq().combine_first(df)
    df_resampled = df_resampled.interpolate(method='time')
    missing_values = df_resampled.isnull().sum().sum()
    if missing_values > 0:
        print(f'{missing_values} missing values, using fallback methods.')
        df_resampled = df_resampled.bfill().ffill()

    return df_resampled

def forwardfill_missing_values(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    #df_resampled = df.resample(interval).ffill()
    df_resampled = df.resample(interval).asfreq().combine_first(df).ffill()
    missing_values = df_resampled.isnull().sum().sum()
    if missing_values > 0:
        print(f'{missing_values} missing values, using fallback methods.')
        df_resampled = df_resampled.bfill()
    return df_resampled

def backfill_missing_values(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    #df_resampled = df.resample(interval).bfill()
    df_resampled = df.resample(interval).asfreq().combine_first(df).bfill()
    missing_values = df_resampled.isnull().sum().sum()
    if missing_values > 0:
        print(f'{missing_values} missing values, using fallback methods.')
        df_resampled = df_resampled.ffill()
    return df_resampled

def fill_missing_values_with_constant(df: pd.DataFrame, interval: str, method: str, const: int = 0) -> pd.DataFrame:
    #df_resampled = df.resample(interval).asfreq()
    df_resampled = df.resample(interval).asfreq().combine_first(df)

    if method == 'const':
        df_resampled =  df_resampled.fillna(const)
    elif method == 'min':
        df_resampled = df_resampled.fillna(df.min())
    else:
        df_resampled = df_resampled.fillna(df.mean())

    missing_values = df_resampled.isnull().sum().sum()
    if missing_values > 0:
        print(f'{missing_values} missing values, using fallback methods.')
        df_resampled = df_resampled.bfill()
    
    return df_resampled

def resample_to_even_intervals(df: pd.DataFrame, sampling_rate: float, method: str, const: int = 0) -> pd.DataFrame:
    """
    Resamples the data to even intervals based on the chosen interval and method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the time column and data.
        sampling_rate (float): Desired resampling interval (e.g., '100ms', '1s').
        method (str): Resampling method to use:
            - 'interpolate': Gradual value change from last to next available sample.
            - 'mean': Fills gaps with the mean of the values.
            - 'min': Fills gaps with the minimum of the values.
            - 'ffill': Forward fills missing values using the last known value.
            - 'bfill': Backward fills missing values using the next known value.
            - 'const' (int): If method 'const' is selected, an integer to fill in missing values with.

    Returns:
        pd.DataFrame: Resampled DataFrame with even intervals.
    """
    df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')

    interval = str(sampling_rate) + 's'

    # Sets the time column as the index
    df.set_index('time', inplace=True)

    # Remove duplicate timestamps
    df = df.loc[~df.index.duplicated(keep='first')]

    if method == 'interpolate':
        df_resampled = interpolate_missing_values(df, interval)

    elif method == 'ffill':
        df_resampled = forwardfill_missing_values(df, interval)

    elif method == 'bfill':
        df_resampled = backfill_missing_values(df, interval)
    
    elif method == 'const':
        df_resampled = fill_missing_values_with_constant(df, interval, method, const)

    elif method == 'mean':
        df_resampled = fill_missing_values_with_constant(df, interval, method, const)

    elif method == 'min':
        df_resampled = fill_missing_values_with_constant(df, interval, method, const)

    else:
        raise ValueError(f'ERROR: Invalid resampling method "{method}". Choose from "interpolate", "mean", "min", "ffill", "bfill", "const".')

    for col in df.columns:
        if col != 'time':
            precision = infer_precision(col)
            if precision > 0:
                #print(f'INFO: Setting precision to {precision} for {col}.')
                df_resampled[col] = df_resampled[col].round(precision)
            else:
                print(f'WARNING: No precision specified for {col}, setting to 2.')
                df_resampled[col] = df_resampled[col].round(2)
        
    # Resets the index and renames it back to 'time'
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'time'}, inplace=True)

    df_resampled = convert_timestamps_from_localized_datetime_to_miliseconds(df_resampled, 'time')
    
    return df_resampled

def process_file_in_batches(file_path: Path, output_path: Path, sampling_rate: float, chunk_size: int, method: str, const: int) -> None:
    """
    Processes a large CSV file in chunks to improve memory efficiency.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed output CSV file.
        chunk_size (int): Number of rows to process at a time.
        method (str): Resampling method ('interpolate', 'mean', 'min', 'ffill', 'bfill', 'const').
    """
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    processed_chunks = []

    for chunk in chunks:
        if not chunk.empty:
            chunk_resampled = resample_to_even_intervals(chunk, sampling_rate, method, const)
            processed_chunks.append(chunk_resampled)
        else:
            print(f'Chunk empty, skipping.')

    final_df = pd.concat(processed_chunks, ignore_index=True)
    save_pandas_dataframe_to_csv(final_df, output_path)
    print(f'Processed and saved: {output_path}')

if __name__ == '__main__':
    base_path = get_base_path()

    # Change before running
    dataset_path = base_path / 'Raw_relevant'
    method = 'const'
    constant = 0
    chunk_size = 2000
    manual_sampling_rate = True
    sampling_rate = 1   # 1s

    resampled_dataset_path = base_path / f'Sync_ready'
    files = get_all_csv_files_in_directory(dataset_path)
    no_files = len(files)
    start_ts = time.time()

    counter = 1
    for file in files:
        filename = file.name
        print(f'Processing file {counter}/{no_files} {filename}...')

        df = read_csv_to_pandas_dataframe(file)

        if not manual_sampling_rate:
            sampling_rate = infer_expected_sampling_rate(filename)
        if sampling_rate >= 0:
            print(f'INFO: Setting sampling rate to {sampling_rate}s for {filename}.')
            if chunk_size >= 0:
                file_out = resampled_dataset_path / filename
                process_file_in_batches(file, file_out, sampling_rate, chunk_size, method, constant)
            else:
                print(f'WARNING: Invalid chunk size, skipping.')
        else:
            print(f'WARNING: Could not set sampling rate, skipping.')
        counter = counter + 1
    
    end_ts = time.time()
    total_t = end_ts - start_ts
    print(f'The process took {total_t:.6f} seconds.')