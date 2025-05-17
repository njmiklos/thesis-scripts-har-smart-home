"""
This script computes statistical summaries (mean and standard deviation) 
for each sensor measurement grouped by annotation class.

Each input CSV file should include a 'time' column and an 'annotation' column, 
along with measurement columns (e.g., temperature, humidity). Device and sensor 
information is inferred from the filename (e.g., 'd1_temp.csv').

Example output:
```
annotation,d1 humidity mean,d1 temperature mean,d1 humidity std,d1 temperature std
sleeping,60.84192495922353,19.9875153119813,2.025295885965855,0.568817859560363
getting up,60.15740438726107,20.06085094551685,1.073246186435458,0.5688105129562113
preparing breakfast,60.488353056066174,20.13107476114225,1.4386784031872946,0.5533294874269475
```

Environment Configuration:
- Set `ANNOTATIONS_FILE_PATH`, `INPUTS_PATH`, and `OUTPUTS_PATH` in your `.env` file.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
import pandas as pd

from pathlib import Path
from typing import List

from utils.get_env import get_path_from_env
from utils.file_handler import (get_all_csv_files_in_directory, read_csv_to_dataframe, 
                                save_dataframe_to_csv, check_if_output_directory_exists)


def extract_unique_classes(annotations_df: pd.DataFrame) -> List[str]:
    """
    Extracts unique annotation class labels from the annotation file.

    Args:
        annotations_df (pd.DataFrame): The annotations dataframe containing an 'annotation' column.

    Returns:
        List[str]: A list of unique annotation class labels.
    """
    return annotations_df['annotation'].unique()

def aggregate_stats_by_class(df_og: pd.DataFrame, stat: str = '') -> pd.DataFrame:
    """
    Aggregates data by class using mean or standard deviation.

    Args:
        df_og (pd.DataFrame): The input dataframe containing an 'annotation' column.
        stat (str): The type of aggregation required, 'mean' or 'std'. Empty string defaults to 'mean'.

    Returns:
        pd.DataFrame: A new dataframe with mean values for each annotation class.
    """
    df = df_og.copy()
    df = df.drop(columns=['time'], errors='ignore')
    if stat == 'std':
        df = df.groupby('annotation').std().reset_index()
    else:
        df = df.groupby('annotation').mean().reset_index()
    return df

def prefix_column_names_with_sensor_metadata(df: pd.DataFrame, file_path: Path, stat: str = '') -> pd.DataFrame:
    """
    Renames dataframe columns by prefixing them with device and sensor information derived from the file name.

    Args:
        df (pd.DataFrame): The input dataframe with columns to rename.
        file_path (Path): The file path of the data source, used to extract device and sensor names.
        stat (str): The type of aggregation required, 'mean', 'std', ''.

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
            if stat == '':
                new_column_names[col] = f'{device} {sensor} {col}'
            else:
                new_column_names[col] = f'{device} {sensor} {col} {stat}'

    df = df.rename(columns=new_column_names)

    return df

def summarize_class_stats_from_file(file_path: Path) -> pd.DataFrame:
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
    df_mean = aggregate_stats_by_class(df_og, 'mean')
    df_mean = prefix_column_names_with_sensor_metadata(df_mean, file_path, 'mean')

    df_std = aggregate_stats_by_class(df_og, 'std')
    df_std = prefix_column_names_with_sensor_metadata(df_std, file_path, 'std')

    df = pd.merge(df_mean, df_std, on=['annotation'], how='outer')
    return df

def summarize_class_stats_from_dataset(annotations_path: Path, inputs_dir: Path, output_path: Path):
    """
    Processes all CSV files in a directory and combines them into a single summary DataFrame.

    Args:
        annotations_path (Path): The file containing annotations.
        input_dir (Path): Directory containing input CSV files.
        output_path (Path): File path where the summary will be saved.
    """
    annotations_df = read_csv_to_dataframe(annotations_path)
    classes = extract_unique_classes(annotations_df)

    df = pd.DataFrame(classes, columns=['annotation'])

    file_paths = get_all_csv_files_in_directory(inputs_dir)
    if file_paths:
        no_files = len(file_paths)
        counter = 1
        for file_path in file_paths:
            print(f'INFO: Processing file {counter}/{no_files} {file_path.name}...')
            sub_df = summarize_class_stats_from_file(file_path)

            if not sub_df.empty:
                df = pd.merge(df, sub_df, on='annotation', how='left')
            else:
                print(f'WARNING: {file_path.stem} empty, skipping.')
            counter = counter + 1
        
        save_dataframe_to_csv(df, output_path)
        print('INFO: All done.')
    else:
        print('WARNING: No CSV files found. Exiting.')


if __name__ == '__main__':
    output_filename = 'summary.csv'

    annotations_path = get_path_from_env('ANNOTATIONS_FILE_PATH')
    inputs_dir = get_path_from_env('INPUTS_PATH')
    outputs_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(outputs_dir)
    output_path = outputs_dir / output_filename

    summarize_class_stats_from_dataset(annotations_path, inputs_dir, output_path)
