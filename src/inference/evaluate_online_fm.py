"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results. 
The purpose is to test how well an online FM annotated data through an API.

It is recommended to first segment the dataset into episodes. This ensures that less data is loaded into memory at once.
"""
import pandas as pd
import numpy as np
import time

from typing import List, Optional
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from data_processing.compress_measurements import generate_summary
from inference.query_fm_api import get_api_config, send_chat_request, get_system_message, get_request_total_tokens
from inference.evaluation_utils import ClassificationResults, TimeMemoryTracer
from inference.evaluate_ei_model import validate_input, save_to_json_file, visualize_confusion_matrix
from data_processing.annotate_dataset import determine_true_annotation


class ClassificationResultsFM(ClassificationResults):
    """
    Extends 'ClassificationResults' class with 'total_prompt_tokens'.

    Args:
        actual_annotations (Optional[List[str]]): Combined list of all actual annotations.
        predicted_annotations (Optional[List[str]]): Combined list of all predicted annotations.
        max_classification_time_ms (float): The highest classification time (ms) observed.
        max_classification_memory_kb (float): The highest memory usage (kB) observed.
        total_prompt_tokens (int): The total number of tokens needed for the classification.
    """
    def __init__(self, actual_annotations: Optional[List[str]] = None, predicted_annotations: Optional[List[str]] = None,
        max_classification_time_ms: float = 0.0, max_classification_memory_kb: float = 0.0, total_prompt_tokens: int = 0) -> None:
        if total_prompt_tokens < 0:
            raise ValueError(f'Number of tokens must be larger than 0, got {total_prompt_tokens}')
        
        super().__init__(actual_annotations, predicted_annotations, max_classification_time_ms, max_classification_memory_kb)
        self.total_prompt_tokens: int = total_prompt_tokens

    def update(self, other: 'ClassificationResultsFM') -> None:
        """
        Merges another ClassificationResults into this one, in place.
        """
        super().update(other)
        self.total_prompt_tokens += other.total_prompt_tokens

    def generate_report(self, total_classification_time_secs: float) -> dict:
        """
        Generates a performance report from the aggregated classification results.

        Args:
            total_classification_time_secs (float): Total classification time in seconds.

        Returns:
            dict: A report containing:
                - 'classes' (List[str]): A sorted list of unique true and false annotations.
                - 'confusion_matrix' (List[List[int]]): Confusion matrix between true and predicted labels.
                - 'total_no_predictions' (int): Total number of predictions made (i.e., number of windows).
                - 'accuracy' (float): Overall classification accuracy.
                - 'weighted_avg_precision' (float): Weighted average precision.
                - 'weighted_avg_recall' (float): Weighted average recall.
                - 'weighted_avg_f1_score' (float): Weighted average F1 score.
                - 'classification_time_ms' (float): The worst-case classification time (ms).
                - 'peak_memory_kb' (float): The worst-case memory usage (kB),
                - 'total_classification_time_secs' (float): Total classification time in seconds.
                - 'total_prompt_tokens' (int): Total number of tokens needed for the classification.
        """
        report = super().generate_report(total_classification_time_secs)
        report['total_prompt_tokens'] = self.total_prompt_tokens
        return report

def format_window_for_stage_1(df: pd.DataFrame) -> str:
    """
    Generates a compact, human-readable summary of sensor data from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series sensor data.

    Returns:
        str: Summary of compressed sensor values.
    """
    return generate_summary(df)

def get_prompt(stage: int) -> str:
    """
    Reads in input prompt text saved in a text file based on the classification stage (details are in my thesis).

    Args:
        stage (int): Stage of the classification process with the FM.
    
    Returns:
        str: Prompt text.
    """
    if stage == 1:
        filename = 'prompt_stage_1.txt'
    else:
        filename = 'prompt_stage_2.txt'

    with open(filename) as f:
        return f.read()

def describe_window_by_window(df: pd.DataFrame, window_size: int, overlap_size: int, 
    model: str) -> Optional['ClassificationResults']:
    """
    Segments the input DataFrame into overlapping windows, asks the FM to describe changes in the measurements
    in each window, records the answers, the worst-case classification time and memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least an 'annotation' column.
        window_size (int): Number of rows in each window.
        overlap_size (int): Number of rows that overlap between consecutive windows.
        
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
    
    api_config = get_api_config(model)
    
    complete_results = ClassificationResults()

    start_position = 0
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        window = df.iloc[start_position : end_position]

        most_common_annotation = df['annotation'].value_counts().idxmax()
        most_common_annotation = str(most_common_annotation)

        trace = TimeMemoryTracer()
        formatted_window = format_window_for_stage_1(window)
        prompt = get_prompt(1)
        
        system_response = send_chat_request(api_config=api_config, prompt=prompt, user_message=formatted_window)
        description = get_system_message(system_response)
        tokens = get_request_total_tokens(system_response)

        window_classification_time_ms, window_classification_memory_kb = trace.stop()

        window_results = ClassificationResultsFM(
            actual_annotations=[most_common_annotation],
            predicted_annotations=[description],
            max_classification_time_ms=window_classification_time_ms,
            max_classification_memory_kb=window_classification_memory_kb,
            total_prompt_tokens=tokens
        )
        complete_results.update(window_results)

        start_position += (window_size - overlap_size)

    return complete_results

def process_files(window_size: int, window_overlap: int, model_name: str, annotations_file_path: Path, 
                  input_dir_path: Path, output_dir_path: Path) -> None:
    """
    Loads the specified Edge Impulse model, processes every CSV file in the input directory,
    and writes a combined report.

    Args:
        window_size (int): Number of rows per sliding window.
        window_overlap (int): Number of rows to overlap between consecutive windows.
        model_name (str): The model to use for chat completions. Defaults to 'meta-llama-3.1-8b-instruct'.
            Other options (08.04.2025): 'internvl2.5-8b', 'c' (DeepSeek R1), 'deepseek-r1-distill-llama-70b',
            'llama-3.3-70b-instruct'.
        annotations_file_path (Path): Path to the file containing true annotations.
        input_dir_path (Path): Directory containing the input CSV files to process.
        output_dir_path (Path): Directory where the final JSON report will be saved.
    
    Returns:
        None
    """
    annotations_df = read_csv_to_pandas_dataframe(annotations_file_path)

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

        episode_classification_results = describe_window_by_window(episode_df, true_annotation, 
                                                                   window_size, window_overlap, model_name)
        
        if episode_classification_results: # i.e. not skipped
            complete_classification_results.update(episode_classification_results)

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
    model_name = 'llama-3.3-70b-instruct'

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    annotations_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_files(window_size, window_overlap, model_name, annotations_file_path, input_dir_path, output_dir_path)

    # TODO
    # - do stage 1 in parts, save to json
    # - do stage 2 in parts, save classification to json