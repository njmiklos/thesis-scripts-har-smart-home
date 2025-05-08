"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results. 
The purpose is to test EI models locally.

The dataset should be segmented into episodes. This ensures that less data is loaded into memory at once.

This version tests the Multi-Model Approach. Each window is processed by three specialized DL models
at the same time. Each model returns a class and its confidence about the choice. The class with 
the highest confidence is saved as the prediction for the window, and results from all windows
determine the efficiency of the approach.
"""
import pandas as pd
import numpy as np
import time

from typing import List, Optional
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from inference.classify_with_ei_model import load_model, close_loaded_model, classify_window, get_top_prediction
from inference.evaluation_utils import ClassificationResults, TimeMemoryTracer
from inference.evaluate_ei_model import (validate_window_size_and_overlap, save_to_json_file, visualize_confusion_matrix, 
                                         infer_model_name, get_column_set)
from data_processing.annotate_dataset import determine_true_annotation


def get_model_column_indices(loaded_models: dict, df: pd.DataFrame) -> dict:
    """
    Computes the column indices for each model based on its expected input features.

    Args:
        loaded_models (dict): A dictionary mapping model names (e.g., 'food', 'routines') 
                              to loaded Edge Impulse model runners.
        df (pd.DataFrame): The full input DataFrame containing all available columns for an episode.

    Returns:
        dict: A dictionary mapping each model name to a list of column indices (integers) 
              that correspond to the features required by that model. These indices can be 
              used to efficiently extract the required columns from a NumPy array.
    """
    model_column_indices = dict()
    for model_name in loaded_models.keys():
        column_names = get_column_set(model_name)
        indices = []
        for col in column_names:
            indices.append(df.columns.get_loc(col))
        model_column_indices[model_name] = indices
    return model_column_indices

def flatten_window_for_models(model_column_indices: dict, window_values: np.ndarray) -> dict:
    """
    Flattens a single window of raw values into model-specific input vectors.

    Args:
        model_column_indices (dict): Mapping of model names to column indices to extract from window.
        window_values (np.ndarray): The full window data as a NumPy array.

    Returns:
        dict: Mapping of model names to their respective flattened input feature lists.
    """
    formatted_windows = dict()
    for model_name, indices in model_column_indices.items():
        selected_values = window_values[:, indices]
        flattened = selected_values.ravel().astype(float).tolist()
        formatted_windows[model_name] = flattened
    return formatted_windows

def classify_window_all_models(loaded_models: dict, windows: dict):
    """
    Runs classification for a single data window across multiple models.

    Args:
        loaded_models (dict): A dictionary mapping model names to loaded model runners.
        window (dict): A dictionary mapping model names to formatted windows.

    Returns:
        dict: A dictionary mapping model names to their classification results.
    """
    if loaded_models is None:
        raise ValueError('Empty model list.')
    
    results = dict()
    for name, runner in loaded_models.items():
        results[name] = classify_window(runner, windows[name])
        
    return results

def get_top_class_from_top_pair(classification_results: dict) -> str:
    """
    Returns the highest-confidence class prediction from classification results,
    preferring any class but 'other'. If no such prediction exists, falls back to the most confident 
    prediction of the 'other' class.

    Args:
        classification_results (dict): A dictionary mapping model names to classification outputs.

    Returns:
        str: The class label with the highest associated probability, preferring labels other than "other".
    """
    top_predictions = []
    other_predictions = []

    for result in classification_results.values():
        predicted_class, probability = get_top_prediction(result)
        if predicted_class == 'other':
            other_predictions.append((predicted_class, probability))
        else:
            top_predictions.append((predicted_class, probability))

    if top_predictions:
        return max(top_predictions, key=lambda x: x[1])[0]
    return max(other_predictions, key=lambda x: x[1])[0]

def classify_with_sliding_windows(df: pd.DataFrame, true_annotation: str, window_size: int, overlap_size: int, 
                                loaded_models: dict) -> Optional['ClassificationResults']:
    """
    Segments the input DataFrame into overlapping windows, runs classification on each window,
    and records the ground truth, model predictions, and the worst-case classification time
    and memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame with an epipsode data to segment.
        true_annotation (str): True annotation for the episode.
        window_size (int): Number of rows in each window.
        overlap_size (int): Number of rows that overlap between consecutive windows.
        loaded_models (dict): A dictionary with model name and pre-loaded Edge Impulse model runner pairs.

    Returns:
        Optional[ClassificationResults]: Returns None if input validation fails. Otherwise, returns an object with:
            - actual_annotations (List[str]): Ground truth annotation from the last row of each window.
            - predicted_annotations (List[str]): Model's predicted class for each window.
            - max_classification_time_ms (float): Maximum time in milliseconds taken by any single window classification.
            - max_classification_memory_kb (float): Maximum memory in kilobytes used by any single window classification.
    """
    total_rows = len(df)

    input_valid = validate_window_size_and_overlap(total_rows, window_size, overlap_size)
    if not input_valid:
        return None
    
    model_column_indices = get_model_column_indices(loaded_models, df)

    df_values = df.to_numpy()

    complete_results = ClassificationResults()

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        window_values = df_values[start_position:end_position]

        resource_tracker = TimeMemoryTracer()
        flattened_windows = flatten_window_for_models(model_column_indices, window_values)
        classification_results = classify_window_all_models(loaded_models, flattened_windows)
        window_classification_time_ms, window_classification_memory_kb = resource_tracker.stop()

        predicted_annotation = get_top_class_from_top_pair(classification_results)

        window_results = ClassificationResults(
            actual_annotations=[true_annotation],
            predicted_annotations=[predicted_annotation],
            max_classification_time_ms=window_classification_time_ms,
            max_classification_memory_kb=window_classification_memory_kb,
        )
        complete_results.update(window_results)

        start_position += (window_size - overlap_size)

    return complete_results

def load_models(model_file_paths: List[Path]) -> dict:
    """
    Loads multiple Edge Impulse models from the given file paths.

    Args:
        model_file_paths (List[Path]): Filesystem paths to the pre-trained Edge Impulse models.

    Returns:
        dict: A dictionary mapping inferred model types (e.g., 'food', 'routines') to their loaded model runners.
    """
    loaded_models = dict()
    for path in model_file_paths:
        model_name = infer_model_name(path.name)
        runner = load_model(path)
        loaded_models[model_name] = runner

    return loaded_models

def close_loaded_models(models: dict) -> None:
    """
    Closes and cleans up all loaded Edge Impulse models to free system resources.

    Args:
        models (dict): A dictionary mapping model names to loaded model runners.

    Returns:
        None
    """
    for model in models.values():
        close_loaded_model(model)

def process_files(window_size: int, window_overlap: int, model_file_paths: List[Path], annotations_file_path: Path, 
                  input_dir_path: Path, output_dir_path: Path) -> None:
    """
    Loads the specified Edge Impulse model, processes every CSV file in the input directory,
    and writes a combined report.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        model_file_paths (List[Path]): Filesystem paths to the pre-trained Edge Impulse models.
        annotations_file_path (Path): Path to the file containing true annotations.
        input_dir_path (Path): Directory containing the input CSV files to process.
        output_dir_path (Path): Directory where the final JSON report will be saved.
    
    Returns:
        None
    """
    annotations_df = read_csv_to_pandas_dataframe(annotations_file_path)

    loaded_models = load_models(model_file_paths)

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
                                                                       loaded_models)
        
        if episode_classification_results: # i.e. not skipped
            complete_classification_results.update(episode_classification_results)

    close_loaded_models(loaded_models)
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

    model_file_paths = [get_path_from_env('MODEL_PATH_1'), 
                        get_path_from_env('MODEL_PATH_2'), 
                        get_path_from_env('MODEL_PATH_3'),]
    
    annotations_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_files(window_size, window_overlap, model_file_paths, annotations_file_path, input_dir_path, output_dir_path)