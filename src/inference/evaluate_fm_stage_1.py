"""
This code segments a dataset into windows, formats them for classification in stage 2, 
and saves the formatted window into a JSON file.

It is recommended to first segment the dataset into episodes. This ensures that less data is loaded into memory at once.
"""
from typing import List
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from data_processing.annotate_dataset import determine_true_annotation
from inference.query_fm_api import get_api_config, send_chat_request, get_system_message, get_request_total_tokens
from inference.evaluate_ei_model import save_to_json_file

# grab files
# segment every file into windows
# format every window and save the formatted window to json


class Window():
    """
    An object holding data of a window of data.

    Attributes:
        true_annotation (str): True annotations for the window of data.
        data (str): The formatted window or the class predicted for the window.
        processing_time_ms (float): Latency of the model, i.e., time 
            since the object was sent until response arrived.
        max_memory_kb (float): The highest memory usage (kB) observed 
            since the object was sent until response arrived.
        tokens (int): The total number of tokens used for the window at the single stage.
    """
    def __init__(self, true_annotation: str, data: str = '', processing_time_ms: float = 0, 
                 max_memory_kb: float = 0, tokens: int = 0) -> None:
        self.true_annotation: str = true_annotation
        self.data = data
        self.processing_time_ms = processing_time_ms
        self.max_memory_kb = max_memory_kb
        self.tokens = tokens
    
    def to_dictionary(self) -> dict:
        """
        Returns the data about the object as a dictionary.

        Returns:
            dict: A dictionary representation of a Window object.
        """
        return {
            'true_annotation': self.true_annotation,
            'data': self.data,
            'processing_time_ms': self.processing_time_ms,
            'max_memory_kb': self.max_memory_kb,
            'tokens': self.tokens
        }

def convert_Window_list_to_dict_list(windows: List['Window']) -> List[dict]:
    """
    Convert the list of Window objects to a list of dictionaries.

    Args:
        windows (List['Window']): A list of Window objects.
    
    Returns:
        List[dict]: A list of dictionaries representing Window objects.
    """
    return [window.to_dictionary() for window in windows]

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

    files = get_all_csv_files_in_directory(input_dir_path)
    n_files = len(files)

    windows = list()

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
            windows.update(episode_classification_results)

    print(f'Done.')

    windows = convert_Window_list_to_dict_list(windows)
    save_to_json_file(output_dir_path, windows)


if __name__ == '__main__':
    # Parameters to be set
    window_size = 75
    window_overlap = 37
    model_name = 'llama-3.3-70b-instruct'
    prompt = """You are a tomato."""

    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    annotations_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')

    output_dir_path.mkdir(parents=True, exist_ok=True)

    process_files(window_size, window_overlap, model_name, prompt, input_dir_path, output_dir_path)
