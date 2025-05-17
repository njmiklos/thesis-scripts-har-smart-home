"""
This script resamples time-series sensor data to uniform intervals using various gap-filling strategies 
(interpolation, forward-fill, backward-fill, mean, min, or constant values).

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file.
- Refer to `README.md` for full setup and usage instructions.
"""
import pandas as pd
import time

from pathlib import Path

from data_processing.convert_timestamps import (convert_timestamps_from_miliseconds_to_localized_datetime, 
                                                convert_timestamps_from_localized_datetime_to_miliseconds)
from utils.file_handler import (save_dataframe_to_csv, get_all_csv_files_in_directory, check_if_output_directory_exists)
from utils.get_env import get_path_from_env
from data_processing.infer.sensor_metadata import infer_precision, infer_expected_sampling_rate


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

def process_file_in_batches(file_path: Path, output_path: Path, sampling_rate: float, 
                            chunk_size: int, method: str, const: int) -> None:
    """
    Processes a large CSV file in chunks to improve memory efficiency.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed output CSV file.
        sampling_rate (float): Desired resampling interval (e.g., '100ms', '1s').
        chunk_size (int): Number of rows to process at a time.
        method (str): Resampling method ('interpolate', 'mean', 'min', 'ffill', 'bfill', 'const').
        const (int): If method 'const' is selected, an integer to fill in missing values with.
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
    save_dataframe_to_csv(final_df, output_path)
    print(f'Processed and saved: {output_path}')

def resample_files(input_dir: Path, output_dir: Path, method: str, const: int, chunk_size: int, 
                  manual_sampling_rate: bool, sampling_rate: float) -> None:
    """
    Resamples CSV files.

    Args:
        input_dir_path (Path): The directory where the CSV files to be resampled are.
        output_path (Path): The directory where the resampled CSV files will be saved.
        method (str): Resampling method ('interpolate', 'mean', 'min', 'ffill', 'bfill', 'const').
        const (int): If method 'const' is selected, an integer to fill in missing values with.
        chunk_size (int): Number of rows to process at a time.
        sampling_rate (float): Desired resampling interval (e.g., '100ms', '1s').
        manual_sampling_rate (bool): If not set to True, sampling rate will be inffered.
    """
    files = get_all_csv_files_in_directory(input_dir)
    no_files = len(files)
    start_ts = time.time()

    counter = 1
    for file in files:
        filename = file.name
        print(f'Processing file {counter}/{no_files} {filename}...')

        if not manual_sampling_rate:
            sampling_rate = infer_expected_sampling_rate(filename)
        if sampling_rate >= 0:
            print(f'INFO: Setting sampling rate to {sampling_rate}s for {filename}.')
            if chunk_size >= 0:
                file_out = output_dir / filename
                process_file_in_batches(file, file_out, sampling_rate, chunk_size, method, const)
            else:
                print(f'WARNING: Invalid chunk size, skipping.')
        else:
            print(f'WARNING: Could not set sampling rate, skipping.')
        counter = counter + 1
    
    end_ts = time.time()
    total_t = end_ts - start_ts
    print(f'The process took {total_t:.6f} seconds.')


if __name__ == '__main__':
    # Change before running
    method = 'const'
    constant = 0
    chunk_size = 2000
    set_sampling_rate_manually = True
    sampling_rate = 1   # 1s

    input_dir = get_path_from_env('INPUTS_PATH')
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir)

    resample_files(input_dir, output_dir, method, constant, chunk_size, set_sampling_rate_manually, sampling_rate)