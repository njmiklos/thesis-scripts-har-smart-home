from utils.handle_csv import save_pandas_dataframe_to_csv
from utils.get_env import get_base_path

from data_processing.merge_data_dfs import merge_synchronized_files_into_single_df
from data_analysis.visualize_data import generate_heatmap


if __name__ == '__main__':
    base_path = get_base_path()

    # Adjust before running
    input_dataset_path = base_path / 'Synchronized'
    output_database_path = input_dataset_path / 'output'
    output_file = 'synchronized_merged.csv'

    df = merge_synchronized_files_into_single_df(input_dataset_path, skip_unannotated=False)

    if not df.empty:
        success = generate_heatmap(df, output_database_path, 'Correlation Heatmap of All Features Synchronized to 1s')
        if success:
            print(f'Saved a heatmap to {output_database_path}.')
        else:
            print(f'WARNING: Could not save a heatmap.')

        save_pandas_dataframe_to_csv(df, output_database_path / output_file)
        print(f'Saved a table to {output_database_path}/{output_file}.')
    else:
        print('WARNING: DataFrame empty, skipping.')