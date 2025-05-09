"""
This code grabs formatted windows from a JSON file, sends them to an API of a Foundation Model, 
tracks the process, and saves results and metrics into a JSON file.
"""
import json

from pathlib import Path
from typing import List

from utils.get_env import get_path_from_env
from data_processing.compress_for_fm import Window
from inference.evaluation_utils import ClassificationResults
from inference.evaluate_ei_model import save_to_json_file

class ExtendedWindow(Window):
    """
    Extends 'Window' class with an attribute 'tokens' and a function 'update'.

    Attributes:
        true_annotation (str): True annotation for the window of data.
        data (str): The formatted window data or a result of its processing by a model.
        processing_time_ms (float): Total processing time of the window.
        max_memory_kb (float): The highest memory usage (kB) observed during the window processing.
        tokens (int): Total number of tokens used for the window at every stage.
    """
    def __init__(self, true_annotation: str, data: str, processing_time_ms: float = 0, 
                 max_memory_kb: float = 0, tokens: int = 0) -> None:
        super().__init__(true_annotation, data, processing_time_ms, max_memory_kb)

        if tokens < 0:
            raise ValueError(f'Tokens must be larger than 0, got {tokens}.')
        self.tokens = tokens

    def update(self, other: 'ExtendedWindow'):
        """
        Merges two windows' data by creating totals.
        """
        if self.data != other.data:
            self.data = other.data
        self.processing_time_ms += other.processing_time_ms
        if other.max_memory_kb > self.max_memory_kb:
            self.max_memory_kb = other.max_memory_kb
        self.tokens += other.tokens

    def to_dictionary(self) -> dict:
        """
        Returns the data about the object as a dictionary.

        Returns:
            dict: A dictionary representation of a ExtendedWindow object.
        """
        window_dict = super().to_dictionary()
        window_dict['tokens'] = self.tokens
        return window_dict

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
    
def read_from_json_file(input_dir_path: Path) -> dict:
    with open(input_dir_path, 'r') as file:
        return json.load(file)

def validate_window_as_dict(dictionary: dict) -> None:
    """
    Ensures that the dict has all required keys, or raises KeyError.

    Args:
        dictionary (dict): A dictionary representing a Window object.

    Returns:
        None
    """
    required_keys = {'true_annotation', 'data', 'processing_time_ms', 'max_memory_kb'}
    missing = required_keys - dictionary.keys()
    if missing:
        raise KeyError(f'Missing key(s) in window dict: {missing}.')

def convert_dict_list_to_window_list(windows_dict: List[dict]) -> List['ExtendedWindow']:
    """
    Converts a list of dictionaries to a list of Window objects.

    Args:
        windows_dict (List[dict]): A list of dictionaries representing Window objects.
    
    Returns:
        List['ExtendedWindow']: A list of Window objects.
    """
    windows = list()
    for d in windows_dict:
        validate_window_as_dict(d)
        window = ExtendedWindow(d['true_annotation'], d['data'], d['processing_time_ms'], d['max_memory_kb'])
        windows.append(window)
    return windows

def validate_window(window: 'ExtendedWindow'):
    if window.true_annotation == '':
        raise ValueError(f'True annotation cannot be empty.')
    if window.data == '':
        raise ValueError(f'Data cannot be empty.')
    if window.processing_time_ms <= 0:
        raise ValueError(f'Time must be larger than 0, got {window.processing_time_ms} ms.')
    if window.max_memory_kb <= 0:
        raise ValueError(f'Memory must be larger than 0, got {window.max_memory_kb} kb.')

def get_prompt(stage: int, input_dir_path: Path) -> str:
    """
    Reads in input prompt text saved in a text file based on the classification stage (details are in my thesis).

    Args:
        stage (int): Stage of the classification process with the FM.
        input_dir_path (Path): Directory containing the input to process.
    
    Returns:
        str: Prompt text.
    """
    if stage not in (1, 2):
        raise ValueError(f'Stage must be either 1 or 2, got {stage}.')

    filename = f'prompt_stage_{stage}.txt'

    with open(input_dir_path / filename) as f:
        return f.read()
    
def process_windows(model_name: str, stage: int, input_dir_path: Path, output_dir_path: Path) -> None:
    """
    Processes windows saved in a JSON file, and writes a combined result to JSON.

    Args:
        stage (int): Stage of classification with an FM, either 1 or 2. It definies the prompt content.
            Stage 1 summarizes and stage 2 classfies. (Prompt texts are intentionally left out. 
            Please refer to the finished thesis.)
        input_dir_path (Path): Directory containing the input to process.
        output_dir_path (Path): Directory where the final JSON data will be saved.
    
    Returns:
        None
    """
    windows_dict = read_from_json_file(input_dir_path)
    windows = convert_dict_list_to_window_list(windows_dict)

    results = list()

    for counter, window in enumerate(windows, start=1):
        print(f'Working on window {counter}/{len(windows)}...')

        validate_window(window)

        if stage == 1:
            prompt = get_prompt(1, input_dir_path)
            result = summarize_data_with_fm(model_name, prompt, window)
        else:
            prompt = get_prompt(2, input_dir_path)
            result = classify_summary_with_fm(model_name, prompt, window)
        
        results.append(result)

        check_limits(results)

    print(f'Done.')

    results_dict = convert_results_list_to_dict_list(results)
    save_to_json_file(output_dir_path, results_dict, f'windows_stage_{stage}.json')


if __name__ == '__main__':
    stage = 1
    model_name = 'llama-3.3-70b-instruct'

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_windows(model_name, stage, input_dir_path, output_dir_path)