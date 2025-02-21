import pandas as pd

from handle_csv import read_csv_to_pandas_dataframe
from get_env import get_input_path, get_output_path
from visualize_data import generate_timeseries_plot_literal_time


def visualize_columns(df: pd.DataFrame, output_dir_path):
    columns = df.columns
    for column in columns:
        if column != 'Epoch':
            title = f'Training_{str(column)}'
            generate_timeseries_plot_literal_time(df['Epoch'], df[column], title, 'Epoch', str(column), output_dir_path, avg_line=False)


if __name__ == '__main__':
    # Paths
    input_file_name = 'epochs.csv'
    input_file_path = get_input_path() / input_file_name
    output_dir_path = get_output_path()

    output_dir_path.mkdir(parents=True, exist_ok=True)

    training_df = read_csv_to_pandas_dataframe(input_file_path)

    visualize_columns(training_df, output_dir_path)