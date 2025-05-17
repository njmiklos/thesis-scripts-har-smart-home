"""
This script summarizes the results of the annotation process performed by an online Foundation Model via API
fo evaluation. It compares predicted annotations to ground truth labels, tracks resource usage 
(time, memory, token count), and generates a detailed performance report.

The report includes accuracy metrics, a confusion matrix, total processing time, and token usage, 
which helps assess how well the FM performed in an automated annotation task.

Environment Configuration:
- Set paths to input/output folders and the annotations file in your `.env` file.
- Input must be a directory of JSON files containing annotated windows.
- Refer to `README.md` for full setup and usage instructions.
"""
from pathlib import Path
from typing import List, Optional

from utils.get_env import get_path_from_env
from utils.file_handler import (save_to_json_file, load_json_file, get_all_json_files_in_directory, 
                                read_csv_to_dataframe, check_if_output_directory_exists)
from inference.evaluate.utils import ClassificationResults
from inference.process_windows_with_fm import ExtendedWindow, convert_dict_list_to_window_list
from inference.evaluate.ei_model import visualize_confusion_matrix


class ResultsFM(ClassificationResults):
    """
    Extends 'ClassificationResults' class with additional tracking for total time and token usage.

    Args:
        actual_annotations (Optional[List[str]]): Combined list of all actual annotations.
        predicted_annotations (Optional[List[str]]): Combined list of all predicted annotations.
        max_classification_time_ms (float): The highest classification time (ms) observed.
        max_classification_memory_kb (float): The highest memory usage (kB) observed.
        total_time_ms (float): Total time needed for processing of all windows at all stages.
        total_prompt_tokens (int): The total number of tokens needed for processing of all windows at all stages. 
    """
    def __init__(self, actual_annotations: Optional[List[str]] = None, predicted_annotations: Optional[List[str]] = None,
        max_classification_time_ms: float = 0.0, max_classification_memory_kb: float = 0.0, 
        total_time_ms: float = 0, total_prompt_tokens: int = 0) -> None:
        if total_time_ms < 0:
            raise ValueError(f'Time must be at least 0 s, got {total_time_ms} s')
        if total_prompt_tokens < 0:
            raise ValueError(f'Number of tokens must be at least 0, got {total_prompt_tokens}')
        
        super().__init__(actual_annotations, predicted_annotations, max_classification_time_ms, max_classification_memory_kb)
        self.total_time_ms: float = total_time_ms
        self.total_prompt_tokens: int = total_prompt_tokens
    
    def normalize_annotation(self, data: str, annotations_file: Path) -> str:
        """
        Extracts and normalizes a known annotation from a raw data string based on a reference annotation list.
        If multiple annotations are found, the last match is returned.
        If none are found, 'other' is returned.

        Args:
            data (str): The raw annotation string to normalize.
            annotations_file (Path): Path to the file containing true annotations.

        Returns:
            str: The last matching annotation found in the data or 'other'.
        """
        annotations = read_csv_to_dataframe(annotations_file)
        expected_annotations = annotations['annotation'].dropna().astype(str).unique().tolist()
        
        data = data.strip('\n{}').lower()

        found_annotations = [annotation for annotation in expected_annotations if annotation in data]

        if not found_annotations:
            print(f'Did not find annotation in data, saving as \"other\". Data content: {data}')
            return 'other'
        if len(found_annotations) > 1:
            print(f'Found more than 1 annotation in data: {found_annotations}, saving the last one: {found_annotations[-1]}. Data content: {data}')
        return found_annotations[-1]

    def update(self, windows: List['ExtendedWindow'], annotations_file: Path) -> None:
        """
        Updates the classification results based on a list of ExtendedWindow objects.

        Args:
            windows (List[ExtendedWindow]): List of window objects containing prediction and performance data.
            annotations_file (Path): Path to the file containing true annotations.

        Returns:
            None
        """
        self.actual_annotations = list()
        self.predicted_annotations = list()

        for window in windows:
            self.actual_annotations.append(window.true_annotation)
            self.predicted_annotations.append(self.normalize_annotation(window.data, annotations_file))

            if self.max_classification_time_ms < window.processing_time_ms:
                self.max_classification_time_ms = window.processing_time_ms
            
            if self.max_classification_memory_kb < window.max_memory_kb:
                self.max_classification_memory_kb = window.max_memory_kb

            self.total_time_ms += window.processing_time_ms
            self.total_prompt_tokens += window.tokens

    def generate_report(self) -> dict:
        """
        Generates a detailed performance report including confusion matrix, accuracy, and resource usage.

        Returns:
            dict: A dictionary summarizing classification performance and resource usage:
            - 'classes' (List[str]): A sorted list of unique true and false annotations.
            - 'confusion_matrix' (List[List[int]]): Confusion matrix between true and predicted labels.
            - 'total_no_predictions' (int): Total number of predictions made (i.e., number of windows).
            - 'accuracy' (float): Overall classification accuracy.
            - 'weighted_avg_precision' (float): Weighted average precision.
            - 'weighted_avg_recall' (float): Weighted average recall.
            - 'weighted_avg_f1_score' (float): Weighted average F1 score.
            - 'max_classification_time_ms' (float): The worst-case classification time (ms).
            - 'peak_classification_memory_kb' (float): The worst-case memory usage (kB),
            - 'total_classification_time_secs' (float): Total classification time in seconds.
            - 'total_prompt_tokens' (int): The total number of tokens needed for processing of all windows at all stages.
        """
        report = super().generate_report(self.total_time_ms / 1000)
        report['total_prompt_tokens'] = self.total_prompt_tokens
        return report

def validate_window_dict_keys(dictionary: dict) -> None:
    """
    Validates that a dictionary has all required keys to represent an ExtendedWindow object.

    Args:
        dictionary (dict): A dictionary representing an ExtendedWindow object.

    Returns:
        None
    """
    required_keys = {'true_annotation', 'data', 'processing_time_ms', 'max_memory_kb', 'tokens'}
    missing = required_keys - dictionary.keys()
    if missing:
        raise KeyError(f'Missing key(s) in window dict: {missing}.')

def load_windows_from_files(input_dir_path: Path) -> List[ExtendedWindow]:
    """
    Loads all JSON files in a directory and converts them into a list of ExtendedWindow objects.

    Args:
        input_dir_path (Path): Directory containing the JSON files to process.

    Returns:
        List[ExtendedWindow]: A list of ExtendedWindow objects loaded from all JSON files in the directory.
    """
    files = get_all_json_files_in_directory(input_dir_path)
    n_files = len(files)

    total_windows = list()
    for counter, file in enumerate(files, start=1):
        filename = file.name
        print(f'Reading file {counter}/{n_files} {filename}...')

        file_windows_dict = load_json_file(file)
        file_windows = convert_dict_list_to_window_list(file_windows_dict)
        total_windows.extend(file_windows)

    return total_windows

def generate_fm_annotation_report(input_dir_path: Path, annotations_file: Path, 
                                  output_dir_path: Path, report_filename: str) -> None:
    """
    Evaluates FM-generated annotations by comparing them with true annotations and generates a report.

    Args:
        input_dir_path (Path): Path to the directory containing window files.
        annotations_file (Path): Path to the file containing true annotations.
        output_dir_path (Path): Path where the performance report and confusion matrix will be saved.
        report_filename (str): Name of the output file.

    Returns:
        None
    """
    windows = load_windows_from_files(input_dir_path)
    results = ResultsFM()
    results.update(windows, annotations_file)
    report = results.generate_report()
    save_to_json_file(output_dir_path, report, report_filename)
    visualize_confusion_matrix(output_dir_path, report['classes'], report['confusion_matrix'])


if __name__ == '__main__':
    report_filename = 'fm_valuation.json'

    input_path = get_path_from_env('INPUTS_PATH')
    annotations_file = get_path_from_env('ANNOTATIONS_FILE_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir_path)

    generate_fm_annotation_report(input_path, annotations_file, output_dir_path, report_filename)