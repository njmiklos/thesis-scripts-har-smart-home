import json
import numpy as np

from pathlib import Path

from get_env import get_input_path, get_output_path
from visualize_data import generate_confusion_matrix


def load_json_file(file_path: Path) -> dict:
    """
    Loads a JSON file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as file:    
        return json.load(file)
    
def convert_matrix_values_to_percentages(matrix: np.ndarray) -> np.ndarray:
    """
    Convert confusion matrix values to percentages by normalizing each row.

    Args:
        matrix (np.ndarray): Confusion matrix.

    Returns:
        np.ndarray: Normalized confusion matrix in percentage format.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    return (matrix / row_sums) * 100  # Convert to percentage

def visualize_confusion_matrix(data: dict, model_version: str, output_path: Path) -> None:
    """
    Generate a confusion matrix from JSON data and save it as an image.

    Args:
        data (dict): JSON data containing the confusion matrix.
        model_version (str): Model version ("int8" or "float32"), both available in data.
        output_path (Path): Directory path to save the confusion matrix image.
    """
    if model_version not in data['validation']:
        raise ValueError(f"Invalid model_version: {model_version}. Choose from: {list(data['validation'].keys())}")

    conf_matrix = np.array(data['validation'][model_version]['confusion_matrix'])
    class_names = data['validation'][model_version]['class_names']

    conf_matrix_percentage = convert_matrix_values_to_percentages(conf_matrix)

    try:
        success = generate_confusion_matrix(conf_matrix_percentage, class_names, output_path)
        if success:
            print(f'Saved confusion matrix to {output_path}.')
    except Exception as e:
        print(f'Error generating confusion matrix: {e}')


if __name__ == '__main__':
    # Paths
    input_file_name = 'episodic-split-only-annotated-data-validation-metrics.json'
    input_file_path = get_input_path() / input_file_name
    output_dir_path = get_output_path()

    output_dir_path.mkdir(parents=True, exist_ok=True)

    report = load_json_file(input_file_path)
    visualize_confusion_matrix(report, 'float32', output_dir_path)

