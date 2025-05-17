"""
This script filters time series data from CSV files by a specific date and time range, creating visualizations 
to help inspect data patterns over the defined period. Plots are saved as individual image files, one for each column, 
using the filename and column name in the title.

Environment Configuration:
- Set `INPUTS_DIR` and `OUTPUTS_DIR` in your `.env` file to specify the locations of input data and where plots will be saved.
- Input files must be CSVs with a `time` column containing datetime-formatted values.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, get_all_csv_files_in_directory, check_if_output_directory_exists
from data_analysis.visualize.utils import generate_timeseries_plot
from data_processing.filter import filter_by_date, filter_by_time_range


def visualize_data_period(input_dir: Path, output_dir: Path, date: str, time_start: str, time_end: str) -> None:
    """
    Filters time series data from CSV files by a specific date and time range, then generates plots for each 
    numeric column.

    Args:
        input_dir (Path): Directory containing input CSV files.
        output_dir (Path): Directory where generated plots will be saved.
        date (str): Date to filter the data by (format: YYYY-MM-DD).
        time_start (str): Start of the time range (format: HH:MM:SS+TZ).
        time_end (str): End of the time range (format: HH:MM:SS+TZ).

    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir)

    for path in files:
        print(path)

        df = read_csv_to_dataframe(path)
        df = filter_by_date(df, date)
        df = filter_by_time_range(df, time_start, time_end)

        if not df.empty:
            for column in df.columns:
                if column != 'time':
                    title = f'{str(path.stem)} - {column}'
                    success = generate_timeseries_plot(df['time'], df[column], title, 'Date', column, output_dir)

                    if not success:
                        print(f'Failed to save plot: {title}')
        else:
            print(f'Skipping {path.stem}, no data')


if __name__ == '__main__':
    date = '2024-12-07'
    time_start = '07:00:00+01:00'
    time_end = '19:00:00+01:00'

    input_dir = get_path_from_env('INPUTS_DIR')
    output_dir = get_path_from_env('OUTPUTS_DIR')
    check_if_output_directory_exists(output_dir)

    visualize_data_period(input_dir, output_dir, date, time_start, time_end)