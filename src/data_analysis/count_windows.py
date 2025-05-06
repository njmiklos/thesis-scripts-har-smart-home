"""
Returns the number of windows with the specified window size and overlap that a dataset in the specified directory
can be separated into.

Useful for cases where the number of predictions needs to be known ahead of time, e.g., with paid APIs.
"""

import pandas as pd

from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from inference.evaluate_ei_model import validate_input


def count_windows(df: pd.DataFrame, window_size: int, window_overlap: int) -> int:
    """
    Counts how many sliding windows of a given size and overlap can fit into a DataFrame.

    Args:
        df (pd.DataFrame): The data to segment.
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.

    Returns:
        int: The total number of complete windows that can be extracted.
    """
    total_rows = len(df)
    input_valid = validate_input(total_rows, window_size, window_overlap)
    if not input_valid:
        return 0  # Window shorter than the selected window size

    segments = 0
    start_position = 0
    while start_position + window_size <= total_rows:
        segments += 1
        start_position += window_size - window_overlap

    return segments

def process_files(window_size: int, window_overlap: int, input_dir_path: Path) -> None:
    """
    Iterates over all CSV files in a directory and accumulate the total number of windows.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        input_dir_path (Path): Directory containing the input CSV files to process.

    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir_path)
    n_files = len(files)

    total_no_windows = 0

    counter = 1
    for file in files:
        filename = file.name
        print(f'Segmenting file {counter}/{n_files} {filename}...')

        df = read_csv_to_pandas_dataframe(file)
        total_no_windows += count_windows(df, window_size, window_overlap)

        counter = counter + 1

    print(f'Total of {total_no_windows} windows.')


if __name__ == '__main__':
    window_size = 600
    window_overlap = 150

    input_dir_path = get_path_from_env('INPUTS_PATH')

    process_files(window_size, window_overlap, input_dir_path)