"""
This code segments an annotated dataset into windows, classifies them, 
and summarizes the results. The purpose is to test an EI model locally.

It is recommended to first segment the dataset into episodes.
This ensures that less data is loaded into memory at once.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, roc_curve, recall_score
import time
import tracemalloc

from typing import List, Tuple, Optional
from pathlib import Path

from get_env import get_input_path, get_output_path
from handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from classify_eim import load_model, close_loaded_model, classify_window, get_top_prediction


def format_window_for_classification(df: pd.DataFrame, relevant_columns: Optional[List[str]]) -> List[float]:
    """
    Flattens a DataFrame into a single Python list.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data of a single window to be classified.
        relevant_columns (Optional[List[str]]): The list of columns expected by the model, all other are dropped.

    Returns:
        List[Any]: A flattened list of floats to be classified.

    Notes:
        - .values gives a 2D NumPy array
        - .ravel() flattens it
        - .tolist() turns it into a Python list
        - the elements are casted into floats
    """
    if relevant_columns:
        df = df[list(relevant_columns)]
    flattened_features = df.values.ravel().tolist()
    features = [float(f) for f in flattened_features]
    return features

def validate_input(total_rows: int, window_size: int, overlap_size: int) -> bool:
    """
    Validates input against each other.

    Args:
        total_rows (int): The number of rows in the dataset.
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.

    Returns:
        bool: Returns True if the input values are valid, throws a ValueError otherwise.
    """
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

def classify_window_by_window(df: pd.DataFrame, window_size: int, overlap_size: int, model_file_path: Path, relevant_columns: Optional[List[str]]) -> Optional[Tuple[List[str], List[str]]]:
    """
    Segments a DataFrame into overlapping windows and classifies each.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be segmented.
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.
        model_file_path (Path): Path to the .eim model file.
        relevant_columns (Optional[List[str]]): The list of columns expected by the model, all other are dropped.

    Returns:
        Tuple[List[str], List[str]]: If input is valid, a Tuple with actual and predicted annotations for all windows.
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

            formatted_window = format_window_for_classification(window, relevant_columns)
            classification_result = classify_window(loaded_model, formatted_window)
            prediction = get_top_prediction(classification_result)
            predicted_annotations.append(prediction)
            
            start_position += (window_size - overlap_size)
            segment_count += 1

        close_loaded_model(loaded_model)

        return actual_annotations, predicted_annotations

def generate_report(output_dir_path: Path, actual_annotations: List[str], predicted_annotations: List[str], 
                    classes: Tuple[str], total_time_seconds: float, peak_memory_mb: float):
    """
    Generates and saves a report based on classification results.

    Args:
        output_dir_path (Path): Path to the directory where report files are stored.
        actual_annotations (List[str]): List of actual annotation labels.
        predicted_annotations (List[str]): List of predicted annotation labels.
        classes (List[str]): List of all class labels.
        total_time_seconds (float): Time in seconds needed for the classification process.
        peak_memory_mb (float): Peak memory in MB during the classification process.
    """
    c_matrix = confusion_matrix(actual_annotations, predicted_annotations, labels=classes)
    accuracy = accuracy_score(actual_annotations, predicted_annotations)
    #area_under_roc_curve = roc_curve(actual_annotations, predicted_annotations)
    weighted_avg_precision = average_precision_score(actual_annotations, predicted_annotations, average='weighted')
    weighted_avg_recall = recall_score(actual_annotations, predicted_annotations, average='weighted')
    weighted_avg_f1_score = f1_score(actual_annotations, predicted_annotations, average='weighted')

    report = list()
    matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in c_matrix])
    report.append(f'Matrix:\n{matrix_str}')
    report.append(f'Accuracy: {accuracy}')
    report.append(f'Weighted Avg. Precision: {weighted_avg_precision}')
    report.append(f'Weighted Avg. Recall: {weighted_avg_recall}')
    report.append(f'Weighted Avg. F1 Score: {weighted_avg_f1_score}')
    report.append(f'Classification Time (s): {total_time_seconds}')
    report.append(f'Peak Memory (mb): {peak_memory_mb}')

    output_file_name = 'classification_report.txt'
    output_path = output_dir_path / output_file_name

    with open(output_path, 'w') as f:
        for line in report:
            f.write(f'{line}\n')

def process_files(window_size: int, window_overlap: int, model_file_path: Path, classes: Tuple[str], relevant_columns: Optional[List[str]], input_dir_path: Path, output_dir_path) -> None:
    """
    Processes all CSV files in the input directory by segmenting them into windows,
    classifying the windows, and generating a classficiation result in the output directory.

    Args:
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.
        model_file_path (Path): Path to the .eim model file.
        classes (Tuple[str]): Classes meant to be classified by the selected model.
        relevant_columns (Optional[List[str]]): The list of columns expected by the model, all other are dropped.
        input_dir (Path): Directory containing input CSV files.
        output_dir (Path): Directory to write output files.
    
    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir_path)
    predicted_annotations = list()
    actual_annotations = list()

    start_time = time.perf_counter()
    tracemalloc.start()

    for file in files:
        print(f'Processing file {file}...')
        episode_df = read_csv_to_pandas_dataframe(file)
        ep_actual_annotations, ep_predicted_annotations = classify_window_by_window(episode_df, window_size, window_overlap, model_file_path, relevant_columns)
        
        actual_annotations.extend(ep_actual_annotations)
        predicted_annotations.extend(ep_predicted_annotations)

    end_time = time.perf_counter()
    total_time_seconds = end_time - start_time

    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    generate_report(output_dir_path, actual_annotations, predicted_annotations, classes, total_time_seconds, peak_memory_mb)


if __name__ == '__main__':
    # Parameters to be set
    window_size = 75
    window_overlap = 37
    model_file_name = 'single-model-approach-linux-x86_64-v5.eim'

    # Classes
    classes = ('getting up', 'leaving home', 'entering home', 'preparing for bed', 'airing', 'sleeping', 'working', 'working out', 'relaxing', 'preparing breakfast', 'eating breakfast', 'preparing dinner', 'eating dinner', 'preparing supper', 'eating supper', 'preparing a drink', 'preparing a meal', 'eating a meal', 'other')

    # Columns
    relevant_columns = ['kitchen humidity [%]', 'kitchen luminosity [Lux]', 'kitchen magnitude accelerometer [m/s²]', 'kitchen temperature [°C]', 'entrance humidity [%]', 'entrance luminosity [Lux]', 'entrance magnitude accelerometer [m/s²]', 'entrance temperature [°C]', 'living room humidity [%]', 'living room luminosity [Lux]', 'living room magnitude accelerometer [m/s²]', 'living room temperature [°C]', 'living room CO2 [ppm]', 'living room max sound pressure [dB]', 'hour']

    # Paths
    input_dir_path = get_input_path()
    output_dir_path = get_output_path()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_file_path = input_dir_path / model_file_name

    process_files(window_size, window_overlap, model_file_path, classes, relevant_columns, input_dir_path, output_dir_path)