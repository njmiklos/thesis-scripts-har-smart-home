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
        window_dict['processing_time_s'] = window_dict.pop('processing_time_ms') / 1000
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

def convert_window_list_to_dict_list(windows: List['ExtendedWindow']) -> List[dict]:
    """
    Convert the list of Window objects to a list of dictionaries.

    Args:
        windows (List['Window']): A list of Window objects.
    
    Returns:
        List[dict]: A list of dictionaries representing Window objects.
    """
    return [window.to_dictionary() for window in windows]

def save_windows(output_dir_path: Path, windows: List['Window'], windows_per_file: int, stage: int) -> None:
    """
    Saves Windows to JSON. If windows_per_file > 0, writes that many
    windows per file. Otherwise, writes all windows to a single file.

    Args:
        output_dir_path (Path): Directory where the final JSON data will be saved.
        windows (List['Window']): A list of Window objects.
        windows_per_file (int): Number of windows to be saved per file. If 0 is given,
            all windows are saved to the same file.
        stage (int): Stage of classification with an FM, either 1 or 2. It definies the prompt content.
            Stage 1 summarizes and stage 2 classifies. (Prompt texts are intentionally left out. 
            Please refer to the finished thesis.)

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
        save_to_json_file(output_dir_path, dicts, f'windows_stage_{stage}.json')
    
    print(f'Saved {total_windows} window(s) to {file_counter} file(s).')

def process_windows(model_name: str, stage: int, input_dir_path: Path, output_dir_path: Path, windows_per_file: int) -> None:
    """
    Processes windows saved in a JSON file, and writes a combined result to JSON.

    Args:
        stage (int): Stage of classification with an FM, either 1 or 2. It definies the prompt content.
            Stage 1 summarizes and stage 2 classifies. (Prompt texts are intentionally left out. 
            Please refer to the finished thesis.)
        input_dir_path (Path): Directory containing the input to process.
        output_dir_path (Path): Directory where the final JSON data will be saved.
        windows_per_file (int): Number of windows to be saved per file. If 0 is given,
            all windows are saved to the same file.
    
    Returns:
        None
    """
    windows_dict = load_json_file(input_dir_path)
    windows = convert_dict_list_to_window_list(windows_dict)
    limits_at_end = dict()

    if stage == 1:
        prompt = get_prompt(1, input_dir_path)
    else:
        prompt = get_prompt(2, input_dir_path)
    
    for counter, window in enumerate(windows, start=1):
        print(f'Working on window {counter}/{len(windows)}...')
        resource_tracker = TimeMemoryTracer()

        validate_window(window)

        response = send_chat_request(model=model_name, prompt=prompt, user_message=window.data)

        _, max_memory_kb = resource_tracker.stop()

        latency_ms = response.elapsed.total_seconds() * 1000
        response_json = response.json()
        system_text = response_json['choices'][0]['message']['content']
        total_tokens = response_json['usage']['total_tokens']
        limits_at_end = get_rate_limits(system_text)

        window_update = ExtendedWindow(window.true_annotation, system_text, latency_ms, max_memory_kb, total_tokens)
        window.update(window_update)

    print(f'Done. Remaining rate limits:')
    for key, value in limits_at_end.items():
        print(f'- {key}: {value}')

    save_windows(output_dir_path, windows, windows_per_file, stage)


if __name__ == '__main__':
    stage = 1
    model_name = 'llama-3.3-70b-instruct'
    windows_per_file = 90    # If 0 is given, all windows are saved to the same file.

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_windows(model_name, stage, input_dir_path, output_dir_path, windows_per_file)