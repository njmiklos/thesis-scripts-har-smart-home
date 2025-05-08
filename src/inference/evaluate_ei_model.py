"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results. 
The purpose is to test EI models locally.

The dataset should be segmented into episodes. This ensures that less data is loaded into memory at once.

This version tests one of four DL models (Single, Rooutines, Transitions, Food) at a time, 
each on their unique testing sets.
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
from data_processing.annotate_dataset import determine_true_annotation
from data_analysis.visualize_ei_report import convert_matrix_values_to_percentages
from data_analysis.visualize_data import generate_confusion_matrix


def get_model_column_indices(model_name: str, df: pd.DataFrame) -> List[int]:
    """
    Returns the column indices needed for a specific model.

    Args:
        model_name (str): One of 'single', 'transitions', 'routines', 'food'.
        df (pd.DataFrame): The full input DataFrame for the episode.

    Returns:
        List[int]: Indices of the columns used by the specified model.
    """
    column_names = get_column_set(model_name)
    indices = []
    for col in column_names:
        indices.append(df.columns.get_loc(col))
    return indices

def get_column_set(model: str) -> List[str]:
    """
    Returns the list of columns expected by the specified model.

    Args:
        model (str): One of 'single', 'transitions', 'routines', 'food'.
    
    Returns:
        List[str]: The expected columns for that model.
    """
    column_sets = {
        'single' : ['kitchen humidity [%]', 'kitchen luminosity [Lux]', 'kitchen magnitude accelerometer [m/s²]', 
                    'kitchen temperature [°C]', 'entrance humidity [%]', 'entrance luminosity [Lux]', 
                    'entrance magnitude accelerometer [m/s²]', 'entrance temperature [°C]', 'living room humidity [%]', 
                    'living room luminosity [Lux]', 'living room magnitude accelerometer [m/s²]', 'living room temperature [°C]',
                    'living room CO2 [ppm]', 'living room max sound pressure [dB]', 'hour'],
        'transitions': ['kitchen humidity [%]', 'kitchen luminosity [Lux]', 'kitchen temperature [°C]', 'entrance humidity [%]', 
                        'entrance luminosity [Lux]', 'entrance magnitude accelerometer [m/s²]', 
                        'entrance magnitude gyroscope [°/s]', 'entrance temperature [°C]', 'living room luminosity [Lux]', 
                        'living room temperature [°C]', 'living room air quality index', 'living room CO2 [ppm]', 
                        'living room min sound pressure [dB]', 'hour'],
        'routines' : ['entrance humidity [%]', 'entrance luminosity [Lux]', 'entrance temperature [°C]', 
                      'living room humidity [%]', 'living room luminosity [Lux]', 'living room magnitude accelerometer [m/s²]', 
                      'living room magnitude gyroscope [°/s]', 'living room temperature [°C]', 'living room air quality index', 
                      'living room CO2 [ppm]', 'living room min sound pressure [dB]', 'hour'],
        'food':  ['kitchen humidity [%]', 'kitchen luminosity [Lux]', 'kitchen magnitude accelerometer [m/s²]', 
                  'kitchen magnitude gyroscope [°/s]', 'kitchen temperature [°C]', 'living room humidity [%]', 
                  'living room luminosity [Lux]', 'living room magnitude accelerometer [m/s²]', 
                  'living room magnitude gyroscope [°/s]', 'living room temperature [°C]', 'living room air quality index', 
                  'living room CO2 [ppm]', 'living room min sound pressure [dB]', 'living room max sound pressure [dB]', 'hour']
    }

    if model not in column_sets:
        raise ValueError(f'Cannot return columns, no such model as {model}.')
    return column_sets[model]

def select_columns(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Keeps only the columns needed for 'model'. Raises a ValueError if any required columns are missing.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data of a single window to be classified.
        model (str): The model type, e.g., 'single', 'transitions'. The remaining columns depend on it.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    required_columns = list(get_column_set(model))

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns for model '{model}': {missing_columns}")

    return df[required_columns]

def flatten_window_for_model(window_values: np.ndarray, column_indices: List[int]) -> List[float]:
    """
    Flattens a window's raw values to a model-compatible input list.

    Args:
        window_values (np.ndarray): 2D NumPy array representing the data window.
        column_indices (List[int]): Column indices to include for the model.

    Returns:
        List[float]: Flattened list of floats for model input.

    Notes:
        An EI model accepts data of a window as a continous list of comma separated values
        (i.e., every rows of values ends with a comma and the next row starts in the same line).

        In this code:
        - .values gives a 2D NumPy array
        - .ravel() flattens it
        - .tolist() turns it into a Python list
        - the elements are casted into floats
    """
    selected = window_values[:, column_indices]
    return selected.ravel().astype(float).tolist()

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

def classify_with_sliding_windows(df: pd.DataFrame, annotation: str, window_size: int, overlap_size: int, 
    loaded_model: ImpulseRunner, model_name: str) -> Optional['ClassificationResults']:
    """
    Segments the input DataFrame into overlapping windows, runs classification on each window,
    and records the ground truth, model predictions, and the worst-case classification time
    and memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame with an epipsode data to segment.
        annotation (str): True annotation for the episode.
        window_size (int): Number of rows in each window.
        overlap_size (int): Number of rows that overlap between consecutive windows.
        loaded_model (ImpulseRunner): Pre-loaded Edge Impulse model runner used for inference.
        model_name (str): One of 'single', 'transitions', 'routines', 'food'.

    Returns:
        Optional[ClassificationResults]: Returns None if input validation fails. Otherwise, returns an object with:
            - actual_annotations (List[str]): Ground truth annotation from the last row of each window.
            - predicted_annotations (List[str]): Model's predicted class for each window.
            - max_classification_time_ms (float): Maximum time in milliseconds taken by any single window classification.
            - max_classification_memory_kb (float): Maximum memory in kilobytes used by any single window classification.
    """
    total_rows = len(df)

    input_valid = validate_input(total_rows, window_size, overlap_size)
    if not input_valid:
        return None

    column_indices = get_model_column_indices(model_name, df)
    df_values = df.to_numpy()

    complete_results = ClassificationResults()

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        window_values = df_values[start_position:end_position]

        resource_tracker = TimeMemoryTracer()
        flattened_window = flatten_window_for_model(window_values, column_indices)
        classification_result = classify_window(loaded_model, flattened_window)
        window_classification_time_ms, window_classification_memory_kb = resource_tracker.stop()

        predicted_annotation, _ = get_top_prediction(classification_result)

        window_results = ClassificationResults(
            actual_annotations=[annotation],
            predicted_annotations=[predicted_annotation],
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

def infer_model_name(model_file_name: str) -> str:
    """
    Infers which of the known model name is encoded in the file name.

    Args:
        model_file_name (str): e.g. '2025-05-05-single-v2.zip'

    Returns:
        str: One of 'single', 'transitions', 'routines', or 'food'.

    Raises:
        ValueError: If no known model name can be found in the filename.
    """
    known_names = {'single', 'transitions', 'routines', 'food'}
    for part in model_file_name.split('-'):
        if part in known_names:
            return part
    raise ValueError(f'Could not infer model type from {model_file_name}')

def process_files(window_size: int, window_overlap: int, model_file_path: Path, annotations_file_path: Path, 
                  input_dir_path: Path, output_dir_path: Path) -> None:
    """
    Loads the specified Edge Impulse model, processes every CSV file in the input directory,
    and writes a combined report.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        model_file_path (Path): Filesystem path to the pre-trained Edge Impulse model.
        annotations_file_path (Path): Path to the file containing true annotations.
        input_dir_path (Path): Directory containing the input CSV files to process.
        output_dir_path (Path): Directory where the final JSON report will be saved.
    
    Returns:
        None
    """
    annotations_df = read_csv_to_pandas_dataframe(annotations_file_path)

    loaded_model = load_model(model_file_path)
    model_name = infer_model_name(model_file_path.name)

    files = get_all_csv_files_in_directory(input_dir_path)
    n_files = len(files)

    complete_classification_results = ClassificationResults()

    start_time_in_secs = time.perf_counter()
    for counter, file in enumerate(files, start=1):
        filename = file.name
        print(f'Segmenting and classifying file {counter}/{n_files} {filename}...')

        episode_df = read_csv_to_pandas_dataframe(file)

        last_timestamp = episode_df['time'].iloc[-1]
        true_annotation = determine_true_annotation(annotations_df, last_timestamp)

        episode_classification_results = classify_with_sliding_windows(episode_df, true_annotation, 
                                                                       window_size, window_overlap,
                                                                       loaded_model, model_name)
        
        if episode_classification_results: # i.e. not skipped
            complete_classification_results.update(episode_classification_results)

    close_loaded_model(loaded_model)
    print(f'Done.')

    end_time_in_secs = time.perf_counter()
    total_classification_time_secs = end_time_in_secs - start_time_in_secs

    report = complete_classification_results.generate_report(total_classification_time_secs)
    save_to_json_file(output_dir_path, report)
    visualize_confusion_matrix(output_dir_path, report['classes'], report['confusion_matrix'])


if __name__ == '__main__':
    # Parameters
    window_size = 75
    window_overlap = 37

    # Paths to adjust per device
    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')

    # Paths to adjust per model
    model_file_path = get_path_from_env('MODEL_PATH')
    annotations_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_files(window_size, window_overlap, model_file_path, annotations_file_path, input_dir_path, output_dir_path)