"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results. 
The purpose is to test an EI model locally.

It is recommended to first segment the dataset into episodes. This ensures that less data is loaded into memory at once.

The data is expected to contain only those columns that were used in training, 
except for the timestamp column and the annotation column.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, roc_curve, recall_score
import time
import tracemalloc
import json

from typing import List, Tuple, Optional
from pathlib import Path

from edge_impulse_runner import ImpulseRunner
from get_env import get_input_path, get_output_path
from handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from classify_eim import load_model, close_loaded_model, classify_window, get_top_prediction


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

def start_tracing_time_and_memory() -> float:
    """
    Starts timing and memory tracing.

    Returns:
        The start timestamp (time.perf_counter()).
    """
    tracemalloc.start()
    return time.perf_counter()

def stop_trace(start_time: float) -> Tuple[float, float]:
    """
    Stops timing and memory tracing, and reports the elapsed time and peak memory.

    Args:
        start_time: The timestamp returned by start_trace().

    Returns:
        Tuple[float, float]:
          - elapsed_ms: milliseconds elapsed since start_time
          - peak_kb: peak memory (in KB) during the trace
    """
    end_time = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = (end_time - start_time) * 1000
    peak_kb = peak_bytes / 1024
    return elapsed_ms, peak_kb

def classify_window_by_window(df: pd.DataFrame, window_size: int, overlap_size: int, 
    loaded_model: ImpulseRunner) -> Optional[dict]:
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
    
    predicted_annotations = list()
    actual_annotations = list()

    max_classification_time_ms = 0.0
    max_classification_memory_kb = 0.0

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size

        window = df.iloc[start_position : end_position]

        last_annotation = window['annotation'].iloc[-1]
        actual_annotations.append(last_annotation)

        trace_start = start_tracing_time_and_memory()

        formatted_window = format_window_for_classification(window)
        classification_result = classify_window(loaded_model, formatted_window)
        
        window_classification_time_ms, window_classification_memory_kb = stop_trace(trace_start)
        max_classification_time_ms = max(window_classification_time_ms, max_classification_time_ms)
        max_classification_memory_kb = max(window_classification_memory_kb, max_classification_memory_kb)

        prediction_class, _ = get_top_prediction(classification_result)
        predicted_annotations.append(prediction_class)

        start_position += (window_size - overlap_size)

    return {
        'actual_annotations': actual_annotations,
        'predicted_annotations': predicted_annotations,
        'max_classification_time_ms': max_classification_time_ms,
        'max_classification_memory_kb': max_classification_memory_kb
    }

def save_report_to_json_file(output_dir_path: Path, report: dict, output_file_name: str = 'classification_report.json'):
    """
    Saves the given report dictionary to a JSON file.

    Args:
        output_dir_path (Path): Path to the directory where report files are stored.
        report (dict): The report containing all metrics and confusion matrix.
        output_file_name (Optional[str]): Filename for the report, defaults to 'classification_report.json'.
    """
    output_path = output_dir_path / output_file_name
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

def generate_report(results: dict) -> dict:
    """
    Generates a performance report from the aggregated classification results.

    Args:
        results (dict): Dictionary containing the aggregated outputs from the dataset, with keys:
            - 'actual_annotations' (List[str]): All ground-truth labels collected.
            - 'predicted_annotations' (List[str]): All model predictions collected.
            - 'max_classification_time_ms' (float): Maximum time (ms) taken by any window classification.
            - 'max_classification_memory_kb' (float): Peak memory usage (kB) of any window classification.

    Returns:
        dict: A report containing:
            - 'confusion_matrix' (List[List[int]]): Confusion matrix between true and predicted labels.
            - 'accuracy' (float): Overall classification accuracy.
            - 'weighted_avg_recall' (float): Weighted average recall.
            - 'weighted_avg_f1_score' (float): Weighted average F1 score.
            - 'classification_time_ms' (float): The worst-case classification time (ms).
            - 'peak_memory_kb' (float): The worst-case memory usage (kB).
    """
    actual_annotations = results['actual_annotations']
    predicted_annotations = results['predicted_annotations']
    max_classification_time_ms = results['max_classification_time_ms']
    max_classification_memory_kb = results['max_classification_memory_kb']

    classes = list(set(actual_annotations))
    c_matrix = confusion_matrix(actual_annotations, predicted_annotations, labels=classes)
    accuracy = accuracy_score(actual_annotations, predicted_annotations)
    #area_under_roc_curve = roc_curve(actual_annotations, predicted_annotations)
    #weighted_avg_precision = average_precision_score(actual_annotations, predicted_annotations, average='weighted')
    weighted_avg_recall = recall_score(actual_annotations, predicted_annotations, average='weighted')
    weighted_avg_f1 = f1_score(actual_annotations, predicted_annotations, average='weighted')

    return {
        'confusion_matrix': c_matrix.tolist(),
        'accuracy': accuracy,
        'weighted_avg_recall': weighted_avg_recall,
        'weighted_avg_f1': weighted_avg_f1,
        'max_classification_time_ms': max_classification_time_ms,
        'peak_classification_memory_kb': max_classification_memory_kb
    }

def update_results(complete_results: dict, episode_classification_results: dict) -> dict:
    """
    Merges a single episode's classification results into the running aggregate results.

    Args:
        complete_results (dict): Aggregate results so far, with keys:
            - 'actual_annotations' (List[str]): combined list of all actual annotations
            - 'predicted_annotations' (List[str]): combined list of all predicted annotations
            - 'max_classification_time_ms' (float): the highest classification time (ms) observed
            - 'max_classification_memory_kb' (float): the highest memory usage (kB) observed
        episode_classification_results (dict): One episode's results:
            - 'actual_annotations' (List[str]): list of actual annotations within the episode
            - 'predicted_annotations' (List[str]): list of predicted annotations within the episode
            - 'max_classification_time_ms' (float): the highest classification time (ms) observed within the episode
            - 'max_classification_memory_kb' (float): the highest memory usage (kB) observed within the episode

    Returns:
        dict: Updated aggregate results containing:
            - 'actual_annotations': combined list of all actual annotations
            - 'predicted_annotations': combined list of all predicted annotations
            - 'max_classification_time_ms': the highest classification time (ms) observed
            - 'max_classification_memory_kb': the highest memory usage (kB) observed
    """
    episode_actual_annotations = episode_classification_results['actual_annotations']
    episode_predicted_annotations = episode_classification_results['predicted_annotations']
    complete_results['actual_annotations'].extend(episode_actual_annotations)
    complete_results['predicted_annotations'].extend(episode_predicted_annotations)

    episode_max_classification_time_ms = episode_classification_results['max_classification_time_ms']
    episode_max_classification_memory_kb = episode_classification_results['max_classification_memory_kb']
    max_classification_time_ms = max(episode_max_classification_time_ms, complete_results['max_classification_time_ms'])
    max_classification_memory_kb = max(episode_max_classification_memory_kb, complete_results['max_classification_time_ms'])

    return {
        'actual_annotations': complete_results['actual_annotations'], 
        'predicted_annotations': complete_results['predicted_annotations'], 
        'max_classification_time_ms': max_classification_time_ms, 
        'max_classification_memory_kb': max_classification_memory_kb
    }

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
    files = get_all_csv_files_in_directory(input_dir_path)

    complete_results = {
        'actual_annotations': list(), 
        'predicted_annotations': list(), 
        'max_classification_time_ms': 0.0, 
        'max_classification_memory_kb': 0.0
    }

    loaded_model = load_model(model_file_path)

    for file in files:
        print(f'Processing file {file}...')
        episode_df = read_csv_to_pandas_dataframe(file)
        episode_classification_results = classify_window_by_window(episode_df, window_size, window_overlap, loaded_model)
        
        if episode_classification_results: # i.e. not skipped
            complete_results = update_results(complete_results, episode_classification_results)

    close_loaded_model(loaded_model)

    report = generate_report(complete_results)
    save_report_to_json_file(output_dir_path, report)


if __name__ == '__main__':
    # Parameters to be set
    window_size = 75
    window_overlap = 37
    model_file_name = 'single-model-approach-linux-x86_64-v5.eim'

    # Paths
    input_dir_path = get_input_path()
    output_dir_path = get_output_path()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_file_path = input_dir_path / model_file_name

    process_files(window_size, window_overlap, model_file_path, input_dir_path, output_dir_path)