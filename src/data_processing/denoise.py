import pandas as pd
from pathlib import Path

from utils.handle_csv import get_all_csv_files_in_directory, save_pandas_dataframe_to_csv
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

def process_file_in_chunks(file_path: Path, window: int, chunk_size=2000):
    """
    Processes a CSV file in chunks, applies smoothing, and saves the denoised data.

    Parameters:
    - file_path (Path): Path to the CSV file to be processed.
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

        file_dnsn_path = dataset_dnsd_path / file_path.name
        save_pandas_dataframe_to_csv(dnsn_df, file_dnsn_path)
        print(f'INFO: Saved denoised file: {file_dnsn_path}')
    
    except Exception as e:
        print(f'ERROR: Failed to process {file_path}. Reason: {e}')


if __name__ == '__main__':
    base_path = get_path_from_env('BASE_PATH')

    # Set before running
    dataset_org_path = base_path / f'Raw_relevant_resampled_0'
    dataset_dnsd_path = base_path / f'Raw_relevant_resampled_0_denoised_window_5'
    window_size_motion_data = 5
    window_size_ambient_data = 5
    chunk_size = 2000

    files = get_all_csv_files_in_directory(dataset_org_path)
    no_files = len(files)
    counter = 1
    for file in files:
        print(f'INFO: Processing {counter}/{no_files} {file}...')
        window = set_window_size(file, window_size_motion_data, window_size_ambient_data)
        process_file_in_chunks(file, window, chunk_size)
        counter = counter + 1