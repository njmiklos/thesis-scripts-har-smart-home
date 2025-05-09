from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, get_all_csv_files_in_directory
from data_analysis.visualize.utils import generate_timeseries_plot
from data_processing.filter import filter_by_date, filter_by_time_range


if __name__ == '__main__':
    base_path = get_path_from_env('BASE_PATH')
    data_path = base_path / 'Synchronized'
    plot_output_path = data_path / 'Graphs_one_day'

    # Read the data
    files = get_all_csv_files_in_directory(data_path)

    for path in files:
        print(path)

        df = read_csv_to_dataframe(path)
        df = filter_by_date(df, '2024-12-07')
        df = filter_by_time_range(df, '07:00:00+01:00', '19:00:00+01:00')

        if not df.empty:
            # Plot the data
            for column in df.columns:
                if column != 'time':
                    title = f'{str(path.stem)} - {column}'
                    success = generate_timeseries_plot(df['time'], df[column], title, 'Date', column, plot_output_path)

                    if not success:
                        print(f'Failed to save plot: {title}')
        else:
            print(f'Skipping {path.stem}, no data')