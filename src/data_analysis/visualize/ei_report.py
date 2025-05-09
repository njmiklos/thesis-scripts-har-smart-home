"""
This file is used to visualize:
- a training process based on a formatted log report downloaded from Edge Impulse.
- a confusion matrix based on a JSON classification report downloaded from Edge Impulse.
"""
import numpy as np

from pathlib import Path

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, load_json_file
from data_analysis.visualize.utils import generate_confusion_matrix, generate_comparative_timeseries_plot


def visualize_training_process(training_process_file_path: Path, output_dir_path: Path) -> None:
    """
    Visualizes the changes in training and validation accuracy and loss over the epochs.

    Args:
        training_process_file_path (Path): Path to the CSV file containing training metrics, extracted
            from the training log. The format must be like in the following example:
                Epoch,Loss,Accuracy,Validation Loss,Validation Accuracy
                1,0.9531,0.8597,0.5308,0.8722
                2,0.4738,0.8813,0.3968,0.8923
                3,0.3688,0.8963,0.3293,0.9016
        output_dir_path (Path): Directory where the visualizations will be saved.

    Returns:
        None
    """
    training_df = read_csv_to_dataframe(training_process_file_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Accuracy'], training_df['Validation Accuracy'], 
                                         'Training Accuracy', 'Validation Accuracy', 'Model Training - Accuracy Over Time', 'Epoch',
                                         'Accuracy', output_dir_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Loss'], training_df['Validation Loss'], 
                                         'Training Loss', 'Validation Loss', 'Model Training - Loss Over Time', 'Epoch',
                                         'Loss', output_dir_path)
    
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

def visualize_confusion_matrix(data: dict[str, dict], model_version: str, output_path: Path) -> None:
    """
    Generate a confusion matrix from JSON data and save it as an image.

    Args:
       data (dict): A dictionary containing model version keys (e.g., "float32", "int8") with their corresponding evaluation data.
        model_version (str): Model version ("int8" or "float32"), both available in data.
        output_path (Path): Directory path to save the confusion matrix image.
    """
    if model_version not in data:
        raise ValueError(f'Invalid model_version: {model_version}. Choose from: {list(data.keys())}')

    conf_matrix = np.array(data[model_version]['confusion_matrix'])
    class_names = data[model_version]['class_names']

    conf_matrix_percentage = convert_matrix_values_to_percentages(conf_matrix)

    try:
        success = generate_confusion_matrix(conf_matrix_percentage, class_names, output_path)
        if success:
            print(f'Saved confusion matrix to {output_path}.')
    except Exception as e:
        print(f'Error generating confusion matrix: {e}')


if __name__ == '__main__':
    # Paths
    classification_report_file_name = 'model-routines-testing-results.json'
    training_process_file_name = 'epochs.csv'

    classification_report_path = get_path_from_env('INPUTS_PATH') / classification_report_file_name
    training_process_file_path = get_path_from_env('INPUTS_PATH') / training_process_file_name
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    report = load_json_file(classification_report_path)

    #visualize_confusion_matrix(report['validation'], 'float32', output_dir_path)
    visualize_confusion_matrix(report['test'], 'float32', output_dir_path)

    visualize_training_process(training_process_file_path, output_dir_path)
