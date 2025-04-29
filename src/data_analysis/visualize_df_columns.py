from utils.get_env import get_base_path
from utils.handle_csv import read_csv_to_pandas_dataframe
from data_analysis.visualize_data import generate_timeseries_plot, generate_scatter_plot
from data_processing.filter_df import filter_by_date


if __name__ == '__main__':
    base_path = get_base_path()
    database_path = base_path / 'Synchronized'
    database_file_path = database_path / 'synchronized_merged_selected.csv'
    output_path_whole_period_graphs = database_path / 'Graphs_whole_period'
    output_path_day_graphs = database_path / 'Graphs_one_day'

    df = read_csv_to_pandas_dataframe(database_file_path)
    
    if not df.empty:

        # Whole period graphs
        for column in df.columns:
            if column != 'time':
                success = generate_scatter_plot(df['time'], df[column], f"{column}", 'Date', column, output_path_whole_period_graphs)
                if not success:
                    print(f'Failed to save plot: {column}')

        # Day graphs
        df = filter_by_date(df, '2024-12-07')
        for column in df.columns:
            if column != 'time':
                success = generate_timeseries_plot(df['time'], df[column], f"{column}", 'Date', column, output_path_day_graphs)
                if not success:
                    print(f'Failed to save plot: {column}')

    else:
        print(f'Skipping {database_path}, no data')