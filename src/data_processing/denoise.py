"""
This script denoises time-series sensor data by applying a Simple Moving Average (SMA) to each column 
(excluding timestamps). It supports chunked processing for memory efficiency and can clip extreme values 
based on statistical thresholds.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file to specify input and output directories.
- Configure `window_size_ambient_data`, `window_size_motion_data`, and `chunk_size` as needed.
- Refer to `README.md` for full setup and usage instructions.
"""
import pandas as pd
from pathlib import Path

from utils.file_handler import get_all_csv_files_in_directory, save_dataframe_to_csv, check_if_directory_exists
from utils.get_env import get_path_from_env
from data_processing.infer.sensor_metadata import infer_precision
from data_analysis.report_utils import calculate_thresholds


def get_simple_moving_average(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """
    Applies a Simple Moving Average (SMA) to smooth data while maintaining original precision.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - col (str): Column name to apply the SMA.
    - window (int): Number of periods for calculating the moving average.

    Returns:
    - pd.DataFrame: DataFrame with the smoothed column.
    """
    df[col] = df[col].rolling(window=window, min_periods=1).mean()
    precision = infer_precision(col)
    df[col] = df[col].round(precision)
    return df

def clip_data_above_extreme_thresholds(chunk: pd.DataFrame, df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Clips values in a DataFrame column beyond an extreme threshold based on standard deviations.

    Parameters:
    - chunk (pd.DataFrame): DataFrame chunk being processed.
    - df (pd.DataFrame): Full DataFrame used for computing mean and standard deviation.
    - col (str): Column name to clip values.

    Returns:
    - pd.DataFrame: DataFrame with clipped values in the specified column.
    """
    col_mean = df[col].mean()
    col_std_dev = df[col].std()
    lower_threshold, upper_threshold = calculate_thresholds(df, col, col_mean, col_std_dev, 3)
    chunk[col] = chunk[col].clip(lower_threshold, upper_threshold)
    return chunk

def set_window_size(file_path: Path, window_motion: int, window_ambient: int):
    """
    Determines the appropriate window size based on the type of data in the file name.

    Parameters:
    - file_path (Path): Path to the file being processed.
    - window_motion (int): Window size for motion data.
    - window_ambient (int): Window size for ambient data.

    Returns:
    - int: The selected window size.
    """
    if 'motion' in str(file_path.name):
        window = window_motion
    else:
        window = window_ambient
    return window

def process_file_in_chunks(file_path: Path, output_dir: Path, window: int, chunk_size=2000):
    """
    Processes a CSV file in chunks, applies smoothing, and saves the denoised data.

    Parameters:
    - file_path (Path): Path to the CSV file to be processed.
    - output_path (Path): The directory where the denoised CSV files will be saved.
    - window (int): Window size for the smoothing function.
    - chunk_size (int, optional): Number of rows per chunk. Default is 2000.

    Returns:
    - None
    """
    try:     
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        processed_chunks = []

        for chunk in chunks:
            if not chunk.empty:
                for column in chunk.columns:
                    if column != 'time':
                        chunk = get_simple_moving_average(chunk, column, window)

                processed_chunks.append(chunk)
            else:
                print(f'WARNING: Chunk empty, skipping.')

        dnsn_df = pd.concat(processed_chunks, ignore_index=True)

        file_dnsn_path = output_dir / file_path.name
        save_dataframe_to_csv(dnsn_df, file_dnsn_path)
        print(f'INFO: Saved denoised file: {file_dnsn_path}')
    
    except Exception as e:
        print(f'ERROR: Failed to process {file_path}. Reason: {e}')

def denoise_files(input_dir: Path, output_dir: Path, window_size_ambient_data: int, window_size_motion_data: int, 
                  chunk_size=2000) -> None:
    """
    Denoises CSV files.

    Args:
        input_dir_path (Path): The directory where the CSV files to be denoised are.
        output_path (Path): The directory where the denoised CSV files will be saved.
        window_size_ambient_data, window_size_motion_data (int): Window size for the smoothing function.
            The correct size is inffered from the data type provided.
        chunk_size (int, optional): Number of rows per chunk. Default is 2000.

    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir)
    no_files = len(files)
    counter = 1
    for file in files:
        print(f'INFO: Processing {counter}/{no_files} {file}...')
        window = set_window_size(file, window_size_motion_data, window_size_ambient_data)
        process_file_in_chunks(file, output_dir, window, chunk_size)
        counter = counter + 1


if __name__ == '__main__':
    window_size_motion_data = 5
    window_size_ambient_data = 5
    chunk_size = 2000

    input_dir = get_path_from_env('INPUTS_PATH')
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_directory_exists(output_dir)

    denoise_files(input_dir, output_dir, window_size_ambient_data, window_size_motion_data, chunk_size)
