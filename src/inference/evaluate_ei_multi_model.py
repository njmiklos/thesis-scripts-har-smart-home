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
import time

from typing import List, Optional
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from inference.classify_with_ei_model import load_model, close_loaded_model, classify_window, get_top_prediction
from inference.evaluation_utils import ClassificationResults, TimeMemoryTracer
from inference.evaluate_ei_model import (format_window_for_classification, validate_input, save_to_json_file, 
                                         visualize_confusion_matrix, infer_model_type)
from data_processing.annotate_dataset import determine_annotation


def formatt_window_for_models(model_names: List[str], window: pd.DataFrame) ->  dict:
    """
    Flattens a DataFrame into a single Python list.

    Args:
        model_names (List[str]): One of 'single', 'transitions', 'routines', 'food'.
        window (pd.DataFrame): Data for one window.

    Returns:
        (dict): A dictionary mapping model names to formatted windows.
    """
    formatted_windows = dict()
    for name in model_names:
        formatted_windows[name] = format_window_for_classification(window, name)
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

def classify_window_by_window(df: pd.DataFrame, annotation: str, window_size: int, overlap_size: int, 
                                loaded_models: dict) -> Optional['ClassificationResults']:
    """
    Segments the input DataFrame into overlapping windows, runs classification on each window,
    and records the ground truth, model predictions, and the worst-case classification time
    and memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least an 'annotation' column.
        annotation (str): True annotation for the episode.
        window_size (int): Number of rows in each window.
        overlap_size (int): Number of rows that overlap between consecutive windows.
        loaded_models (dict): A dictionary with model name and pre-loaded Edge Impulse model runner pairs.

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

        trace = TimeMemoryTracer()

        formatted_windows = formatt_window_for_models(loaded_models.keys(), window)

        classification_results = classify_window_all_models(loaded_models, formatted_windows)
        window_classification_time_ms, window_classification_memory_kb = trace.stop()

        predicted_annotation = get_top_class_from_top_pair(classification_results)
        window_results = ClassificationResults(
            actual_annotations=[annotation],
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
        type = infer_model_type(path.name)
        runner = load_model(path)
        loaded_models[type] = runner

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
    counter = 1

    start_time_in_secs = time.perf_counter()
    for file in files:
        filename = file.name
        print(f'Segmenting and classifying file {counter}/{n_files} {filename}...')

        episode_df = read_csv_to_pandas_dataframe(file)

        last_timestamp = episode_df['time'].iloc[-1]
        annotation = determine_annotation(annotations_df, last_timestamp)

        episode_classification_results = classify_window_by_window(episode_df, annotation, 
                                                                   window_size, window_overlap, 
                                                                   loaded_models)
        
        if episode_classification_results: # i.e. not skipped
            complete_classification_results.update(episode_classification_results)

        counter = counter + 1

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