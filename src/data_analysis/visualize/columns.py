"""
This script visualizes time series data from a synchronized dataset by generating plots for each available column. 
It supports filtering by a specific date or plotting all available data using either line plots or scatter plots.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file to define the input data directory and output location for plots.
- Input data must be a CSV file with a `time` column and one or more numeric measurement columns.
- Output visualizations are saved as individual plot files, one per column.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, check_if_output_directory_exists
from data_analysis.visualize.utils import generate_timeseries_plot, generate_scatter_plot
from data_processing.filter import filter_by_date


def visualize_columns_from_day(day: str, input_file_path: Path, output_dir: Path) -> None:
    """
    Filters and visualizes data for a specific day using time series line plots.

    Args:
        day (str): Day to visualize columns from, in the format 'YYYY-MM-DD'.
        input_file_path (Path): Path to the CSV file containing the synchronized data.
        output_dir (Path): Directory where the generated plots will be saved.

    Returns:
        None
    """
    df = read_csv_to_dataframe(input_file_path)
    if df is not None and not df.empty:
        df = filter_by_date(df, day)
        for column in df.columns:
            if column != 'time':
                success = generate_timeseries_plot(df['time'], df[column], f'{column}', 'Date', column, output_dir)
                if not success:
                    print(f'Failed to save plot: {column}')
    else:
        print(f'Skipping {input_file_path}, no data')

def visualize_columns(input_file_path: Path, output_dir: Path) -> None:
    """
    Visualizes all columns in the dataset using scatter plots without filtering by date.

    Args:
        input_file_path (Path): Path to the CSV file containing the synchronized data.
        output_dir (Path): Directory where the generated plots will be saved.

    Returns:
        None
    """
    df = read_csv_to_dataframe(input_file_path)
    
    if df is not None and not df.empty:
        for column in df.columns:
            if column != 'time':
                success = generate_scatter_plot(df['time'], df[column], f'{column}', 'Date', column, output_dir)
                if not success:
                    print(f'Failed to save plot: {column}')
    else:
        print(f'Skipping {input_file_path}, no data')


if __name__ == '__main__':
    input_filename = 'synchronized_merged_selected.csv'
    day = '2024-12-07'

    input_dir = get_path_from_env('INPUTS_PATH')
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir)
    database_file_path = input_dir / input_filename

    visualize_columns_from_day(day, database_file_path, output_dir)
    visualize_columns(database_file_path, output_dir)