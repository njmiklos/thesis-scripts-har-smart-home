"""
This code segments an annotated dataset into windows, classifies them, and summarizes the results
to test a model.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix

from typing import List, Tuple, Any
from pathlib import Path

from get_env import get_input_path, get_output_path
from handle_csv import read_csv_to_pandas_dataframe
from classify_eim import load_model, close_loaded_model, classify_window


def init_confusion_matrix(actual_annotations: List[str], predicted_annotations: List[str], classes: List[str]):
    confusion_matrix = confusion_matrix(actual_annotations, predicted_annotations, labels=classes)
    return confusion_matrix

def format_window_for_classification(df: pd.DataFrame) -> List[Any]:
    """
    Flattens a DataFrame into a single Python list.

    - .values gives a 2D NumPy array
    - .ravel() flattens it
    - .tolist() turns it into a Python list

    Args:
        df (pd.DataFrame): The input DataFrame containing the data of a single window to be classified.

    Returns:
        List[Any]: A flattened list of integers and floats to be classified.
    """
    return df.values.ravel().tolist()

def validate_input(total_rows: int, window_size: int, overlap_size: int) -> bool:
    if total_rows == 0:
        raise ValueError('The input DataFrame is empty. There is no data to segment.')
    
    if window_size > total_rows:
        raise ValueError(f'Window size {window_size} is larger than total number of input rows {total_rows}.')
    
    if window_size < 1:
        raise ValueError(f'Invalid window size: {window_size}. A segment must contain at least one row.')
    
    if overlap_size < 0:
        raise ValueError(f'Invalid overlap size: {overlap_size}. Overlap must be 0 or greater.')

    if overlap_size >= window_size:
        raise ValueError(f'Invalid overlap size: {overlap_size}. The overlap must be smaller than the window size ({window_size}).')

    return True

def classify_window_by_window(df: pd.DataFrame, window_size: int, overlap_size: int, model_file_path: Path) -> Tuple[List[str], List[str]]:
    """
    Segments a DataFrame into overlapping windows and classifies each.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be segmented.
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.
        model_file_path (Path): Path to the .eim model file.

    Returns:
        List[str]: A list of predicted labels for all windows. # TODO
    """
    total_rows = len(df)

    input_valid = validate_input(total_rows, window_size, overlap_size)
    if input_valid:
        predicted_annotations = list()
        actual_annotations = list()

        loaded_model = load_model(model_file_path)

        start_position = 0
        segment_count = 1
        while start_position + window_size <= total_rows:
            end_position = start_position + window_size

            window = df.iloc[start_position : end_position]

            last_annotation = window['annotation'].iloc[-1]
            actual_annotations.append(last_annotation)

            formatted_window = format_window_for_classification(window)
            classification_result = classify_window(loaded_model, formatted_window)
            prediction = get_single_best_prediction(classification_result)
            predicted_annotations.append(prediction)
            
            start_position += (window_size - overlap_size)
            segment_count += 1

        close_loaded_model(loaded_model)

        return actual_annotations, predicted_annotations


if __name__ == '__main__':
    window_size = 300
    window_overlap = 150

    # Paths
    input_dir_path = get_input_path()
    output_dir_path = get_output_path()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_file_path = input_dir_path / 'single-model-approach-linux-x86_64-v5.eim'

    dataset_file_name = 'synchronized_merged_selected_annotated_new_col_names.csv'
    input_dataset_path = input_dir_path / dataset_file_name

    # grab all episode files in directory
    # for every episode file, classify as many windows from every file
    episode_df = read_csv_to_pandas_dataframe(input_dataset_path)

    actual_annotations, predicted_annotations = classify_window_by_window(episode_df, window_size, window_overlap, model_file_path)