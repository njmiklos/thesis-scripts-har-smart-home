"""
This script processes pre-formatted windows of data by sending them to a Foundation Model API 
for annotation or summarization. It tracks processing time and memory usage, manages token statistics, 
and writes the new window data and metrics to a JSON file.

Environment Configuration:
- Define API credentials and I/O directory paths in your `.env` file.
- Provide input data in JSON format in the input folder.
- Provide a text file with a prompt in a text file.
- Refer to `README.md` for full setup instructions.
"""
from pathlib import Path
from typing import List

from utils.get_env import get_path_from_env
from utils.file_handler import save_to_json_file, load_json_file, check_if_output_directory_exists
from data_processing.compress_for_fm import Window
from inference.query_fm_api import send_chat_request, get_rate_limits
from inference.evaluate.utils import TimeMemoryTracer


class ExtendedWindow(Window):
    """
    Extends 'Window' class with an attribute 'tokens' and a function 'update'.

    Attributes:
        true_annotation (str): True annotation for the window of data.
        data (str): A summary (stage 1) or an annotation (stage 2) for a window of data.
        processing_time_ms (float): Total processing time of the window in miliseconds.
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
        Updates this window by aggregating values from another window.

        Args:
            other (ExtendedWindow): The window whose values will be merged.
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
    Converts a list of dictionaries into ExtendedWindow objects.

    Args:
        windows_dict (List[dict]): Serialized window data.

    Returns:
        List[ExtendedWindow]: Parsed list of ExtendedWindow objects.
    """
    windows = list()
    for d in windows_dict:
        validate_window_as_dict(d)
        if 'tokens' not in d: # stage 1 -> has not been processed by an FM -> no tokens have been used
            window = ExtendedWindow(d['true_annotation'], d['data'], d['processing_time_ms'], d['max_memory_kb'], 0)
        else:   # stage 2
            window = ExtendedWindow(d['true_annotation'], d['data'], d['processing_time_ms'], d['max_memory_kb'], d['tokens'])
        windows.append(window)
    return windows

def convert_window_list_to_dict_list(windows: List['ExtendedWindow']) -> List[dict]:
    """
    Convert the list of ExtendedWindow objects to a list of dictionaries.

    Args:
        windows (List['ExtendedWindow']): A list of ExtendedWindow objects.
    
    Returns:
        List[dict]: A list of dictionaries representing ExtendedWindow objects.
    """
    return [window.to_dictionary() for window in windows]

def get_and_validate_prompt(input_dir_path: Path, prompt_filename: str) -> str:
    """
    Reads and validates prompt text from a file.

    Args:
        input_dir_path (Path): Path to the directory with the prompt file.
        prompt_filename (str): Filename of the prompt text file.

    Returns:
        str: Prompt text.
    """
    prompt = ''
    with open(input_dir_path / prompt_filename) as f:
        prompt = f.read()
    if prompt == '':
        raise ValueError('Prompt cannot be empty.')
    return prompt

def save_windows(output_dir_path: Path, output_filename: str, windows: List['ExtendedWindow']) -> None:
    """
    Saves ExtendedWindows to a JSON file.

    Args:
        output_dir_path (Path): Directory where the final JSON data will be saved.
        output_filename (str): Output filename.
        windows (List['ExtendedWindow']): A list of ExtendedWindow objects.

    Returns:
        None
    """   
    total_windows = len(windows)
    dicts = convert_window_list_to_dict_list(windows)
    save_to_json_file(output_dir_path, dicts, output_filename)
    
    print(f'Saved {total_windows} window(s) to file {output_dir_path}/{output_filename}')

def process_windows(model_name: str, input_dir_path: Path, output_dir_path: Path, 
                    input_windows_filename: str, prompt_filename: str) -> None:
    """
    Processes windows saved in a JSON file, and writes a combined result to JSON.

    Args:
        model_name (str): Model identifier for the FM API. Some options (08.05.2025): 'internvl2.5-8b', 
        'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 'llama-4-scout-17b-16e-instruct', 
        'gemma-3-27b-it'.
        input_dir_path (Path): Path to the directory with input JSON.
        output_dir_path (Path): Path to save annotated window results.
        input_windows_filename (str): Filename of the input JSON.
        prompt_filename (str): Filename of the prompt text file.

    Returns:
        None
    """
    windows_dict = load_json_file(input_dir_path / input_windows_filename)
    windows = convert_dict_list_to_window_list(windows_dict)
    limits_at_end = dict()
    prompt = get_and_validate_prompt(input_dir_path, prompt_filename)

    for counter, window in enumerate(windows, start=1):
        print(f'Working on window {counter}/{len(windows)}...')
        resource_tracker = TimeMemoryTracer()

        response = send_chat_request(model=model_name, prompt=prompt, user_message=window.data)

        time_ms, max_memory_kb = resource_tracker.stop()

        limits_at_end = get_rate_limits(response)
        response_json = response.json()
        system_text = response_json['choices'][0]['message']['content']
        total_tokens = response_json['usage']['total_tokens']

        window_update = ExtendedWindow(window.true_annotation, system_text, time_ms, max_memory_kb, total_tokens)
        window.update(window_update)

    print(f'Done. Remaining rate limits:')
    for key, value in limits_at_end.items():
        print(f'- {key}: {value}')

    output_filename = input_windows_filename
    save_windows(output_dir_path, output_filename, windows)


if __name__ == '__main__':
    stage = 1
    model_name = 'gemma-3-27b-it'
    input_windows_filename = 'compressed_windows_test.json'
    prompt_filename = f'prompt_stage_{stage}.txt'

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH') / f'stage_{stage}'
    check_if_output_directory_exists(output_dir_path)

    process_windows(model_name, input_dir_path, output_dir_path, input_windows_filename, prompt_filename)