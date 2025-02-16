import pandas as pd

from pathlib import Path

from get_env import get_input_path, get_output_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from infer_metadata import infer_data_collection_days_from_time_column
from filter_df import filter_by_date


def split_df_into_days(df: pd.DataFrame, output_path: Path):
    days = infer_data_collection_days_from_time_column(df['time'])

    for day in days:
        daily_df = filter_by_date(df, day)
        output_file = f'episode_{day}.csv'
        output_file_path = output_path / output_file
        save_pandas_dataframe_to_csv(daily_df, output_file_path)


if __name__ == '__main__':
    database_input_file = 'synchronized_merged_selected_annotated_filtered.csv'
    database_input_file_path = get_input_path() / database_input_file
    database_output_path = get_output_path()

    df = read_csv_to_pandas_dataframe(database_input_file_path)
    split_df_into_days(df, database_output_path)