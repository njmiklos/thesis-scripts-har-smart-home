from pathlib import Path

from handle_csv import read_csv_to_pandas_dataframe
from get_env import get_input_path, get_output_path
from visualize_data import generate_comparative_timeseries_plot
from visualize_classification import load_json_file, visualize_confusion_matrix


def visualize_training_process(training_process_file_path: Path, output_dir_path: Path) -> None:
    """
    Visualizes the changes in training and validation accuracy and loss over the epochs.

    Args:
        training_process_file_path (Path): Path to the CSV file containing training metrics.
        output_dir_path (Path): Directory where the visualizations will be saved.

    Returns:
        None
    """
    training_df = read_csv_to_pandas_dataframe(training_process_file_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Accuracy'], training_df['Validation Accuracy'], 
                                         'Training Accuracy', 'Validation Accuracy', 'Model Training - Accuracy Over Time', 'Epoch',
                                         'Accuracy', output_dir_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Loss'], training_df['Validation Loss'], 
                                         'Training Loss', 'Validation Loss', 'Model Training - Loss Over Time', 'Epoch',
                                         'Loss', output_dir_path)

def visualize_class_classifiction(model_report_file_path: Path, output_dir_path: Path) -> None:
    """
    Generates and saves a confusion matrix visualization for true and predicted labels.

    Args:
        model_report_file_path (Path): Path to the JSON file containing model classification report.
        output_dir_path (Path): Directory where the confusion matrix will be saved.

    Returns:
        None
    """
    report = load_json_file(model_report_file_path)
    visualize_confusion_matrix(report, 'int8', output_dir_path)


if __name__ == '__main__':
    # Paths
    training_process_file_name = 'epochs.csv'
    model_report_file_name = 'model-report.json'

    training_process_file_path = get_input_path() / training_process_file_name
    model_report_file_path = get_input_path() / model_report_file_name

    output_dir_path = get_output_path()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    visualize_training_process(training_process_file_path, output_dir_path)
    visualize_class_classifiction(model_report_file_path, output_dir_path)