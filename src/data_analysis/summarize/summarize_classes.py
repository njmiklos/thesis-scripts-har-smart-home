"""The script aggregates sensor data by annotation class."""

import pandas as pd

from pathlib import Path
from typing import List

from utils.get_env import get_path_from_env
from utils.file_handler import get_all_csv_files_in_directory, read_csv_to_dataframe, save_dataframe_to_csv


def get_classes(df: pd.DataFrame) -> List[str]:
    """
    Extracts unique annotation classes from the given dataframe.

    Args:
        df (pd.DataFrame): The annotations dataframe containing an 'annotation' column.

    Returns:
        List[str]: A list of unique annotation class labels.
    """
    classes = None
    classes = df['annotation'].unique()
    print(f'Classes ({len(classes)}): {classes}\n')
    return classes

def group_values_by_class(df_og: pd.DataFrame, function: str = '') -> pd.DataFrame:
    """
    Groups the dataframe by annotation classes and calculates the mean of other columns.

    Args:
        df_og (pd.DataFrame): The input dataframe containing an 'annotation' column.
        function (str): The type of aggregation required, 'mean' or 'std'. Empty string defaults to 'mean'.

    Returns:
        pd.DataFrame: A new dataframe with mean values for each annotation class.
    """
    df = df_og.copy()
    df = df.drop(columns=['time'], errors='ignore')
    if function == 'std':
        df = df.groupby('annotation').std().reset_index()
    else:
        df = df.groupby('annotation').mean().reset_index()
    return df

def prefix_column_names_with_device_sensor_function(df: pd.DataFrame, file_path: Path, function: str = '') -> pd.DataFrame:
    """
    Renames dataframe columns by prefixing them with device and sensor information derived from the file name.

    Args:
        df (pd.DataFrame): The input dataframe with columns to rename.
        file_path (Path): The file path of the data source, used to extract device and sensor names.
        function (str): The type of aggregation required, 'mean', 'std', ''.

    Returns:
        pd.DataFrame: The dataframe with updated column names.
    """
    file_path = str(file_path.stem)
    device = file_path[0:2]
    sensor = file_path[3:]

    new_column_names = {}

    for col in df.columns:
        if col == 'annotation':
            new_column_names[col] = col
        elif col == 'time':
            new_column_names[col] = col
        else:
            if function == '':
                new_column_names[col] = f'{device} {sensor} {col}'
            else:
                new_column_names[col] = f'{device} {sensor} {col} {function}'

    df = df.rename(columns=new_column_names)

    return df

def process_file(file_path: Path) -> pd.DataFrame:
    """
    Processes a dataset CSV file to provide a summary of all classes as a DataFrame.
    Every rows is a class, and columns show a mean and a standard deviation of every
    measurement in the class.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The processed dataframe with aggregated values and updated column names.
    """
    df_og = read_csv_to_dataframe(file_path)
    df_mean = group_values_by_class(df_og, 'mean')
    df_mean = prefix_column_names_with_device_sensor_function(df_mean, file_path, 'mean')

    df_std = group_values_by_class(df_og, 'std')
    df_std = prefix_column_names_with_device_sensor_function(df_std, file_path, 'std')

    df = pd.merge(df_mean, df_std, on=['annotation'], how='outer')
    return df


if __name__ == '__main__':
    base_path = get_path_from_env('BASE_PATH')
    dataset_path = base_path / 'Synchronized_annotated'
    annotations_path = base_path / 'annotations_combined.csv'
    summary_path = dataset_path / 'summary_classes_synced.csv'

    annotations_df = read_csv_to_dataframe(annotations_path)
    classes = get_classes(annotations_df)

    df = pd.DataFrame(classes, columns=['annotation']) # creates a dataframe with a single column 'class', where every row is a class

    file_paths = get_all_csv_files_in_directory(dataset_path)

    if file_paths:
        no_files = len(file_paths)
        counter = 1
        for file_path in file_paths:
            print(f'INFO: Processing file {counter}/{no_files} {file_path.name}...')
            sub_df = process_file(file_path)

            if not sub_df.empty:
                df = pd.merge(df, sub_df, on='annotation', how='left')
            else:
                print(f'WARNING: {file_path.stem} empty, skipping.')
            counter = counter + 1
        
        save_dataframe_to_csv(df, summary_path)
        print('INFO: All done.')
    else:
        print('WARNING: No CSV files found. Exiting.')