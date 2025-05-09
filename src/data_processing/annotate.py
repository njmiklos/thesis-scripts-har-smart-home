import pandas as pd
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory


def determine_true_annotation(annotations: pd.DataFrame, time: int) -> str:
    """
    Returns the annotation for the given timestamp based on the annotation intervals.

    Parameters:
        annotations: DataFrame with 'start', 'end', and 'annotation' columns.
        time: Timestamp to annotate.

    Returns:
        The corresponding annotation string, or 'other' if no match.
    """
    for _, row in annotations.iterrows():
        if row['start'] <= time <= row['end']:
            annotation = row['annotation']
            return annotation
    return 'other'

def insert_annotations(df: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates each row in the DataFrame based on the time using the annotations DataFrame.

    Parameters:
        df: The DataFrame to annotate.
        annotations: The DataFrame containing annotation intervals.

    Returns:
        Annotated DataFrame.
    """
    annotations_list = []

    for _, row in df.iterrows():
        time = row['time']
        annotation = determine_true_annotation(annotations, time)
        annotations_list.append(annotation)

    df['annotation'] = annotations_list

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
            print(f'Processing file: {file_path}')

            path_new_file = path_output_dataset / f'{file_stem}.csv'
            df = read_csv_to_pandas_dataframe(file_path)
            df = insert_annotations(df, annotations)
            save_pandas_dataframe_to_csv(df, path_new_file)
            print(f'Annotated file saved: {path_new_file}')


if __name__ == '__main__':
    input_dir_path = get_path_from_env('INPUTS_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    path_annotation_file = input_dir_path / 'annotations_combined.csv'
    path_dataset = input_dir_path / 'files_to_annotate'

    process_data_files(path_dataset, output_dir_path, path_annotation_file)
    print('Done.')

