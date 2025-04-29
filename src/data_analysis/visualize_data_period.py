from utils.get_env import get_base_path
from utils.handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory
from visualize_data import generate_timeseries_plot
from data_processing.filter_df import filter_by_date, filter_by_time_range


if __name__ == '__main__':
    base_path = get_base_path()
    data_path = base_path / 'Synchronized'
    plot_output_path = data_path / 'Graphs_one_day'

    # Read the data
    files = get_all_csv_files_in_directory(data_path)

    for path in files:
        print(path)

        df = read_csv_to_pandas_dataframe(path)
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