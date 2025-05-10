"""
This code grabs formatted windows from a JSON file, sends them to an API of a Foundation Model, 
tracks the process, and saves results and metrics into a JSON file.
"""
from pathlib import Path
from typing import List

from utils.get_env import get_path_from_env
from utils.file_handler import save_to_json_file, load_json_file
from data_processing.compress_for_fm import Window
from inference.query_fm_api import send_chat_request, get_rate_limits
from inference.evaluate.utils import TimeMemoryTracer


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
        Returns the data about the object as a dictionary with processing time in seconds.

        Returns:
            dict: A dictionary representation of a ExtendedWindow object.
        """
        window_dict = super().to_dictionary()
        window_dict['tokens'] = self.tokens
        return window_dict

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
        List['ExtendedWindow']: A list of ExtendedWindow objects.
    """
    windows = list()
    for d in windows_dict:
        validate_window_as_dict(d)
        window = ExtendedWindow(d['true_annotation'], d['data'], d['processing_time_ms'], d['max_memory_kb'])
        windows.append(window)
    return windows

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

def convert_window_list_to_dict_list(windows: List['ExtendedWindow']) -> List[dict]:
    """
    Convert the list of ExtendedWindow objects to a list of dictionaries.

    Args:
        windows (List['ExtendedWindow']): A list of ExtendedWindow objects.
    
    Returns:
        List[dict]: A list of dictionaries representing ExtendedWindow objects.
    """
    return [window.to_dictionary() for window in windows]

def save_windows(output_dir_path: Path, windows: List['ExtendedWindow'], stage: int) -> None:
    """
    Saves ExtendedWindows to a JSON file.

    Args:
        output_dir_path (Path): Directory where the final JSON data will be saved.
        windows (List['ExtendedWindow']): A list of ExtendedWindow objects.
        stage (int): Stage of classification with an FM, either 1 or 2. It definies the prompt content.
            Stage 1 summarizes and stage 2 classifies. (Prompt texts are intentionally left out. 
            Please refer to the finished thesis.)

    Returns:
        None
    """   
    total_windows = len(windows)
    dicts = convert_window_list_to_dict_list(windows)
    output_filename = f'stage_{stage}_windows.json'
    save_to_json_file(output_dir_path, dicts, output_filename)
    
    print(f'Saved {total_windows} window(s) to file {output_dir_path}/{output_filename}')

def process_windows(model_name: str, stage: int, input_dir_path: Path, output_dir_path: Path, 
                    input_windows_filename: str) -> None:
    """
    Processes windows saved in a JSON file, and writes a combined result to JSON.

    Args:
        model_name (str): The model to use for chat completions. Some options (08.05.2025): 'internvl2.5-8b', 
        'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 'llama-4-scout-17b-16e-instruct', 
        'gemma-3-27b-it'.
        stage (int): Stage of classification with an FM, either 1 or 2. It definies the prompt content.
            Stage 1 summarizes and stage 2 classifies. (Prompt texts are intentionally left out. 
            Please refer to the finished thesis.)
        input_dir_path (Path): Directory containing the input to process.
        output_dir_path (Path): Directory where the final JSON data will be saved.
        input_windows_filename (str): Name of JSON file where input windows are.
    
    Returns:
        None
    """
    windows_dict = load_json_file(input_dir_path / input_windows_filename)
    windows = convert_dict_list_to_window_list(windows_dict)
    limits_at_end = dict()

    if stage == 1:
        prompt = get_prompt(1, input_dir_path)
    else:
        prompt = get_prompt(2, input_dir_path)
    
    for counter, window in enumerate(windows, start=1):
        print(f'Working on window {counter}/{len(windows)}...')
        resource_tracker = TimeMemoryTracer()

        response = send_chat_request(model=model_name, prompt=prompt, user_message=window.data)

        _, max_memory_kb = resource_tracker.stop()

        latency_ms = response.elapsed.total_seconds() * 1000
        limits_at_end = get_rate_limits(response)
        response_json = response.json()
        system_text = response_json['choices'][0]['message']['content']
        total_tokens = response_json['usage']['total_tokens']

        window_update = ExtendedWindow(window.true_annotation, system_text, latency_ms, max_memory_kb, total_tokens)
        window.update(window_update)

    print(f'Done. Remaining rate limits:')
    for key, value in limits_at_end.items():
        print(f'- {key}: {value}')

    save_windows(output_dir_path, windows, stage)


if __name__ == '__main__':
    stage = 1
    model_name = 'gemma-3-27b-it'
    input_windows_filename = 'stage_1_windows_test.json'

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_windows(model_name, stage, input_dir_path, output_dir_path, input_windows_filename)