"""
This code segments a dataset into windows, formats them for stage 1, and saves into a JSON file.

It is recommended to first segment the dataset into episodes. This ensures that less data is loaded into memory at once.
"""
import pandas as pd

from typing import List, Optional
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from data_processing.annotate_dataset import determine_true_annotation
from data_processing.compress_measurements import generate_summary
from inference.evaluation_utils import TimeMemoryTracer
from inference.evaluate_ei_model import validate_window_size_and_overlap, save_to_json_file


class Window():
    """
    An object holding data of a window of data.

    Attributes:
        true_annotation (str): True annotation for the window of data.
        data (str): The formatted window data or a result of its processing by a model.
        processing_time_ms (float): Total processing time of the window.
        max_memory_kb (float): The highest memory usage (kB) observed during the window processing.
    """
    def __init__(self, true_annotation: str, data: str, processing_time_ms: float = 0, 
                 max_memory_kb: float = 0) -> None:
        if true_annotation == '':
            raise ValueError(f'True annotation cannot be empty.')
        if data == '':
            raise ValueError(f'Data cannot be empty.')
        if processing_time_ms < 0:
            raise ValueError(f'Time must be larger than 0, got {processing_time_ms} ms.')
        if max_memory_kb < 0:
            raise ValueError(f'Memory must be larger than 0, got {max_memory_kb} kb.')
        
        self.true_annotation: str = true_annotation
        self.data = data
        self.processing_time_ms = processing_time_ms
        self.max_memory_kb = max_memory_kb
    
    def to_dictionary(self) -> dict:
        """
        Returns the data about the object as a dictionary.

        Returns:
            dict: A dictionary representation of a Window object.
        """
        return {
            'true_annotation': self.true_annotation,
            'data': self.data,
            'processing_time_ms': self.processing_time_ms,
            'max_memory_kb': self.max_memory_kb,
        }

def format_window(df: pd.DataFrame) -> str:
    """
    Generates a compact, human-readable summary of sensor data from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series sensor data.

    Returns:
        str: Summary of compressed sensor values.
    """
    return generate_summary(df)

def format_with_sliding_windows(episode_df: pd.DataFrame, annotation: str, window_size: int, 
                                window_overlap: int) -> Optional[List['Window']]:
    """
    Segments the input DataFrame into overlapping windows and processes them for classification
    into a list of Window objects.

    Args:
        episode_df (pd.DataFrame): Input DataFrame with an epipsode data to segment.
        annotation (str): True annotation for the episode.
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.

    Returns:
        Optional[List['Window']]: Returns None if input validation fails. Otherwise, returns a list of Window objects.
    """
    total_rows = len(episode_df)

    input_valid = validate_window_size_and_overlap(total_rows, window_size, window_overlap)
    if not input_valid:
        return None
    
    windows = list()

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        window_df = episode_df.iloc[start_position : end_position]

        resource_tracker = TimeMemoryTracer()
        formatted_window_data = format_window(window_df)
        window_processing_time_ms, window_peak_memory_kb = resource_tracker.stop()

        window_results = Window(annotation, formatted_window_data, window_processing_time_ms, window_peak_memory_kb)
        windows.append(window_results)

        start_position += (window_size - window_overlap)

    return windows

def save_windows(output_dir_path: Path, windows: List['Window'], windows_per_file: int) -> None:
    """
    Saves Windows to JSON. If windows_per_file > 0, writes that many
    windows per file. Otherwise, writes all windows to a single file.

    Args:
        output_dir_path (Path): Directory where the final JSON data will be saved.
        windows (List['Window']): A list of Window objects.
        windows_per_file (int): Number of windows to be saved per file. If 0 is given,
            all windows are saved to the same file.

    Returns:
        None
    """
    if windows_per_file < 0:
        raise ValueError(f'Window number must be ≥ 0, got {windows_per_file}.')
    
    total_windows = len(windows)
    file_counter = 0

    if windows_per_file > 0:
        for start_pos in range(0, total_windows, windows_per_file):
            file_counter += 1
            end_pos = start_pos + windows_per_file
            chunk = windows[start_pos : end_pos]
            chunk_dicts = convert_window_list_to_dict_list(chunk)
            filename = f'compressed_windows_{file_counter}.json'
            save_to_json_file(output_dir_path, chunk_dicts, filename)
    else:
        file_counter = 1
        dicts = convert_window_list_to_dict_list(windows)
        save_to_json_file(output_dir_path, dicts, f'compressed_windows.json')
    
    print(f'Saved {total_windows} window(s) to {file_counter} file(s).')

def convert_window_list_to_dict_list(windows: List['Window']) -> List[dict]:
    """
    Convert the list of Window objects to a list of dictionaries.

    Args:
        windows (List['Window']): A list of Window objects.
    
    Returns:
        List[dict]: A list of dictionaries representing Window objects.
    """
    return [window.to_dictionary() for window in windows]

def process_files(window_size: int, window_overlap: int, annotations_file_path: Path, 
                  input_dir_path: Path, output_dir_path: Path, windows_per_file: int = 0) -> None:
    """
    Processes every CSV file in the input directory, and writes a combined result to JSON.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        annotations_file_path (Path): Path to the file containing true annotations.
        input_dir_path (Path): Directory containing the input CSV files to process.
        output_dir_path (Path): Directory where the final JSON data will be saved.
        windows_per_file (int): Number of windows to be saved per file. If 0 is given,
            all windows are saved to the same file.
    
    Returns:
        None
    """
    annotations_df = read_csv_to_pandas_dataframe(annotations_file_path)

    files = get_all_csv_files_in_directory(input_dir_path)
    n_files = len(files)

    windows = list()

    for counter, file in enumerate(files, start=1):
        filename = file.name
        print(f'Segmenting and formatting {counter}/{n_files} {filename}...')

        episode_df = read_csv_to_pandas_dataframe(file)

        if 'time' not in episode_df.columns:
            raise ValueError(f"Missing 'time' column in file: {filename}")
        
        last_timestamp = episode_df['time'].iloc[-1]
        true_annotation = determine_true_annotation(annotations_df, last_timestamp)

        episode_windows = format_with_sliding_windows(episode_df, true_annotation, window_size, window_overlap)
        
        if episode_windows is not None:
            windows.extend(episode_windows)

    print(f'Done.')

    save_windows(output_dir_path, windows, windows_per_file)


if __name__ == '__main__':
    # Parameters to be set
    window_size = 600
    window_overlap = 198
    windows_per_file = 0    # If 0 is given, all windows are saved to the same file.

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    annotations_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_files(window_size, window_overlap, annotations_file_path, input_dir_path, output_dir_path, windows_per_file)
