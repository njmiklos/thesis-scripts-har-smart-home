import pandas as pd

from typing import Set, List

from get_env import get_base_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime_srs, convert_timestamps_from_localized_datetime_to_miliseconds_srs


def filter_by_date(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the rows from the specified day.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        day (str): The date to filter by in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A DataFrame with rows corresponding to the specified day.
    """
    df['time'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['time'])
    df = df[df['time'].dt.date == pd.to_datetime(day).date()]
    df['time'] = convert_timestamps_from_localized_datetime_to_miliseconds_srs(df['time'])
    return df

def filter_by_time_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only rows within the specified time range.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        start (str): The start time in the format 'HH:MM:SS+TZ'.
        end (str): The end time in the format 'HH:MM:SS+TZ'.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only rows within the specified time range.
    """
    df['time'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['time'])
    df = df[(df['time'].dt.time >= pd.to_datetime(start).time()) & 
        (df['time'].dt.time <= pd.to_datetime(end).time())]
    df['time'] = convert_timestamps_from_localized_datetime_to_miliseconds_srs(df['time'])
    return df

def remove_rows_with_annotation(df: pd.DataFrame, annotation: str) -> pd.DataFrame:
    """
    Removes rows containing the specified annotation.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        annotation (str): The annotation to filter out.

    Returns:
        pd.DataFrame: A DataFrame with rows containing the specified annotation removed.
    """
    df = df[~df['annotation'].str.contains(annotation, na=False)]
    return df

def remove_rows_with_annotations(df: pd.DataFrame, annotations: Set[str]) -> pd.DataFrame:
    """
    Filters out rows containing any of the specified annotations.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        annotations (Set[str]): A set of annotations to filter out.

    Returns:
        pd.DataFrame: A DataFrame with rows containing any of the specified annotations removed.
    """
    for annotation in annotations:
        df = remove_rows_with_annotation(df, annotation)
    return df

def remove_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Removes the specified column from the DataFrame if it exists.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        col_name (str): The name of the column to remove.

    Returns:
        pd.DataFrame: A DataFrame with the specified column removed if it exists.
    """
    df = df.drop(columns=col_name, errors='ignore')
    return df

def remove_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Removes the specified columns from the DataFrame if they exist.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        columns (List[str]): A list of column names to remove.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns removed if they exist.
    """
    for col in columns:
        df = remove_column(df, col)
    return df


if __name__ == '__main__':
    base_path = get_base_path()

    # Annotation, single file
    """
    classes = {'airing', 'preparing for bed', 'sleeping', 'getting up', 'working out', 
        'preparing breakfast', 'eating breakfast', 'preparing dinner', 'eating dinner', 
        'preparing supper', 'eating supper', 'preparing a drink', 'working', 'relaxing', 
        'leaving home', 'entering home', 'preparing a meal', 'eating a meal'}
    transition_activities = {'getting up', 'leaving home', 'entering home', 'preparing for bed', 'airing'}

    # Adjust before running
    dataset_path = base_path / 'Dataset Transition Activities' 
    dataset_file = dataset_path / 'Transition Activities.csv'
    new_dataset_file = dataset_path / 'Transition Activities 2.csv'
    activities_to_filter_out = classes - transition_activities

    df = read_csv_to_pandas_dataframe(dataset_file)
    df = filter_out_rows_containing_annotations(df, activities_to_filter_out)
    save_pandas_dataframe_to_csv(df, new_dataset_file)
    """

    # Filter file
    dataset_path = base_path / 'Synchronized' 
    dataset_file = dataset_path / 'synchronized_merged.csv'
    new_dataset_file = dataset_path / 'synchronized_merged_selected.csv'

    df = read_csv_to_pandas_dataframe(dataset_file)
    columns = ['d1 hygrometer temperature', 'd1 motion_magnitude mag', 'd1 thermometer objectTemperature',
       'd2 hygrometer temperature', 'd2 motion_magnitude mag', 'd2 thermometer objectTemperature',
       'd3 hygrometer temperature', 'd3 motion_magnitude mag', 'd3 thermometer objectTemperature']
    df = remove_columns(df, columns)
    save_pandas_dataframe_to_csv(df, new_dataset_file)

    # Dataset
    """
    files = get_all_csv_files_in_directory(base_path / 'some_dir')

    for path in files:
        print(path)

        df = read_csv_to_pandas_dataframe(path)

        df = filter_by_date(df, '2024-12-08')
        df = filter_by_time_range(df, '08:55:52+01:00', '08:56:00+01:00')

        save_pandas_dataframe_to_csv(df, path)
    """