"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results. 
The purpose is to test an EI model locally.

It is recommended to first segment the dataset into episodes. This ensures that less data is loaded into memory at once.

The data is expected to contain only those columns that were used in training, 
except for the timestamp column and the annotation column.
"""
import pandas as pd
import numpy as np
import time
import json

from typing import List, Optional
from pathlib import Path

from inference.edge_impulse_runner import ImpulseRunner
from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from inference.classify_with_ei_model import load_model, close_loaded_model, classify_window, get_top_prediction
from inference.evaluation_utils import ClassificationResults, TimeMemoryTracer
from data_analysis.visualize_ei_report import convert_matrix_values_to_percentages
from data_analysis.visualize_data import generate_confusion_matrix


def format_window_for_classification(df: pd.DataFrame) -> List[float]:
    """
    Flattens a DataFrame into a single Python list.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data of a single window to be classified.

    Returns:
        List[Any]: A flattened list of floats to be classified.

    Notes:
        An EI model accepts data of a window as a continous list of comma separated values
        (i.e., every rows of values ends with a comma and the next row starts in the same line).

        In this code:
        - .values gives a 2D NumPy array
        - .ravel() flattens it
        - .tolist() turns it into a Python list
        - the elements are casted into floats
    """
    if 'time' in df.columns: 
        df = df.drop(columns=['time'])
    if 'annotation' in df.columns: 
        df = df.drop(columns=['annotation'])

    flattened_features = df.values.ravel().tolist()
    features = [float(f) for f in flattened_features]
    return features

def validate_input(total_rows: int, window_size: int, overlap_size: int) -> bool:
    """
    Validates input against each other.

    Args:
        total_rows (int): The number of rows in the dataset.
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.

    Returns:
        bool: Returns True if the input values are valid, throws a ValueError otherwise.
    """
    if total_rows == 0:
        raise ValueError('The input DataFrame is empty. There is no data to segment.')
    
    if window_size > total_rows:    # Window to be skipped
        return False
    
    if window_size < 1:
        raise ValueError(f'Invalid window size: {window_size}. A segment must contain at least one row.')
    
    if overlap_size < 0:
        raise ValueError(f'Invalid overlap size: {overlap_size}. Overlap must be 0 or greater.')

    if overlap_size >= window_size:
        raise ValueError(f'Invalid overlap size: {overlap_size}. The overlap must be smaller than the window size ({window_size}).')

    return True

def classify_window_by_window(df: pd.DataFrame, window_size: int, overlap_size: int, 
    loaded_model: ImpulseRunner) -> Optional['ClassificationResults']:
    """
    Segments the input DataFrame into overlapping windows, runs classification on each window,
    and records the ground truth, model predictions, and the worst-case classification time
    and memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least an 'annotation' column.
        window_size (int): Number of rows in each window.
        overlap_size (int): Number of rows that overlap between consecutive windows.
        loaded_model (ImpulseRunner): Pre-loaded Edge Impulse model runner used for inference.

    Returns:
        Optional[dict]: Returns None if input validation fails. Otherwise, returns a dict with:
            - actual_annotations (List[str]): Ground truth annotation from the last row of each window.
            - predicted_annotations (List[str]): Model's predicted class for each window.
            - max_classification_time_ms (float): Maximum time in milliseconds taken by any single window classification.
            - max_classification_memory_kb (float): Maximum memory in kilobytes used by any single window classification.
    """
    total_rows = len(df)

    input_valid = validate_input(total_rows, window_size, overlap_size)
    if not input_valid:
        return None
    
    complete_results = ClassificationResults()

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        window = df.iloc[start_position : end_position]

        most_common_annotation = df['annotation'].value_counts().idxmax()
        most_common_annotation = str(most_common_annotation)

        trace = TimeMemoryTracer()
        formatted_window = format_window_for_classification(window)
        classification_result = classify_window(loaded_model, formatted_window)
        window_classification_time_ms, window_classification_memory_kb = trace.stop()

        prediction_class, _ = get_top_prediction(classification_result)

        window_results = ClassificationResults(
            actual_annotations=[most_common_annotation],
            predicted_annotations=[prediction_class],
            max_classification_time_ms=window_classification_time_ms,
            max_classification_memory_kb=window_classification_memory_kb,
        )
        complete_results.update(window_results)

        start_position += (window_size - overlap_size)

    return complete_results

def save_to_json_file(output_dir_path: Path, dictionary: dict, output_file_name: str = 'classification_report.json'):
    """
    Saves the given dictionary to a JSON file.

    Args:
        output_dir_path (Path): Path to the directory where report files are stored.
        dictionary (dict): The dictionary to be saved.
        output_file_name (Optional[str]): Filename for the report, defaults to 'classification_report.json'.
    """
    output_path = output_dir_path / output_file_name
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dictionary, f, indent=4)
    
    print(f'Saved {output_path}.')

def visualize_confusion_matrix(output_dir_path: Path, classes: List[str], confusion_matrix: List[List[int]]) -> None:
    """
    Generate a confusion matrix from JSON data and save it as an image.

    Args:
        classes (List[str]): A sorted list of unique true and false annotations.
        confusion_matrix (List[List[int]]): Confusion matrix between true and predicted labels.
    """
    confusion_matrix_array = np.array(confusion_matrix)
    conf_matrix_percentage = convert_matrix_values_to_percentages(confusion_matrix_array)

    generate_confusion_matrix(conf_matrix_percentage, classes, output_dir_path)

def process_files(window_size: int, window_overlap: int, model_file_path: Path, input_dir_path: Path, output_dir_path) -> None:
    """
    Loads the specified Edge Impulse model, processes every CSV file in the input directory,
    and writes a combined report.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        model_file_path (Path): Filesystem path to the pre-trained Edge Impulse model.
        input_dir_path (Path): Directory containing the input CSV files to process.
        output_dir_path (Path): Directory where the final JSON report will be saved.
    
    Returns:
        None
    """
    start_time_in_secs = time.perf_counter()

    files = get_all_csv_files_in_directory(input_dir_path)
    n_files = len(files)

    complete_classification_results = ClassificationResults()

    loaded_model = load_model(model_file_path)

    counter = 1
    for file in files:
        filename = file.name
        print(f'Segmenting and classifying file {counter}/{n_files} {filename}...')

        episode_df = read_csv_to_pandas_dataframe(file)
        episode_classification_results = classify_window_by_window(episode_df, window_size, window_overlap, loaded_model)
        
        if episode_classification_results: # i.e. not skipped
            complete_classification_results.update(episode_classification_results)

        counter = counter + 1

    close_loaded_model(loaded_model)
    print(f'Done.')

    end_time_in_secs = time.perf_counter()
    total_classification_time_secs = end_time_in_secs - start_time_in_secs

    report = complete_classification_results.generate_report(total_classification_time_secs)
    save_to_json_file(output_dir_path, report)
    visualize_confusion_matrix(output_dir_path, report['classes'], report['confusion_matrix'])


if __name__ == '__main__':
    # Parameters to be set
    window_size = 75
    window_overlap = 37

    # Paths
    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    model_file_name = get_path_from_env('MODEL_PATH')
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_file_path = input_dir_path / model_file_name

    process_files(window_size, window_overlap, model_file_path, input_dir_path, output_dir_path)