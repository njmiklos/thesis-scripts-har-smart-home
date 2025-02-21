import pandas as pd

from handle_csv import read_csv_to_pandas_dataframe
from get_env import get_input_path, get_output_path
from visualize_data import generate_comparative_timeseries_plot


if __name__ == '__main__':
    # Paths
    input_file_name = 'epochs.csv'
    input_file_path = get_input_path() / input_file_name
    output_dir_path = get_output_path()

    output_dir_path.mkdir(parents=True, exist_ok=True)

    training_df = read_csv_to_pandas_dataframe(input_file_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Accuracy'], training_df['Validation Accuracy'], 
                                         'Training Accuracy', 'Validation Accuracy', 'Model Training - Accuracy Over Time', 'Epoch',
                                         'Accuracy', output_dir_path)

    generate_comparative_timeseries_plot(training_df['Epoch'], training_df['Loss'], training_df['Validation Loss'], 
                                         'Training Loss', 'Validation Loss', 'Model Training - Loss Over Time', 'Epoch',
                                         'Loss', output_dir_path)