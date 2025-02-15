import pandas as pd
from pathlib import Path

from get_env import get_base_path
from handle_csv import (read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory)


def insert_annotations(df: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Inserts annotations into the dataset based on time intervals.

    Parameters:
        df: The DataFrame to annotate.
        annotations: The DataFrame containing annotation intervals.

    Returns:
        Annotated DataFrame.
    """
    df['annotation'] = 'other'  # default annotation

    for _, row in annotations.iterrows():
        annotated_start_time = row['start']
        annotated_end_time = row['end']
        annotation = row['annotation']

        mask = (df['time'] >= annotated_start_time) & (df['time'] <= annotated_end_time)
        df.loc[mask, 'annotation'] = annotation
    
    return df

def process_data_files(path_dataset: Path, path_output_dataset: Path, path_annotation_file: Path) -> None:
    """
    Processes all CSV files in the given dataset path, annotates them, 
    and saves them with a new file name.

    Parameters:
        path_dataset: Path to the directory containing data files.
        path_output_dataset: Path to the directory where annotated files are to be saved.
        annotations: Annotations DataFrame.

    Returns:
        None.
    """
    paths_data_files = get_all_csv_files_in_directory(path_dataset)
    annotations = read_csv_to_pandas_dataframe(path_annotation_file)

    for file_path in paths_data_files:
        if file_path != path_annotation_file:
            file_stem = file_path.stem
            print(f"Processing file: {file_path}")

            path_new_file = path_output_dataset / f'{file_stem}.csv'
            df = read_csv_to_pandas_dataframe(file_path)
            df = insert_annotations(df, annotations)
            save_pandas_dataframe_to_csv(df, path_new_file)
            print(f"Annotated file saved: {path_new_file}")

if __name__ == '__main__':
    base_path = get_base_path()
    path_dataset = base_path / 'Synchronized'
    path_output_dataset = base_path / 'Synchronized_annotated'
    path_annotation_file = base_path / 'annotations_combined.csv'

    process_data_files(path_dataset, path_output_dataset, path_annotation_file)
    print('Done.')

