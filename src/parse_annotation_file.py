"""
This files validates a file listing all annotated episodes. 

In the file, every episode is a row. It must include the following columns:
    - `start`: Episode start time (e.g., 1733353200000 – UTC milliseconds, Europe/Berlin timezone).
    - `end`: Episode end time (see the format above).
    - `annotation`: Activity class (e.g., 'sleeping', 'airing').
"""
import pandas as pd

from get_env import get_annotations_file_path
from handle_csv import read_csv_to_pandas_dataframe


def parse_timestamps(annotations: pd.DataFrame) -> None:
    """
    Validates and converts the 'start' and 'end' columns of a DataFrame to datetime objects.

    Args:
        annotations (pd.DataFrame): The input DataFrame containing 'start' and 'end' columns 
        with timestamps in string format.

    Raises:
        ValueError: If any timestamp does not match the format '%Y-%m-%d %H:%M:%S'.
    """
    columns = ['start', 'end']
    for column in columns:
        date = annotations[column]
        try:
            annotations[column] = pd.to_datetime(annotations[column], format='%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format in column '{column}': {e}")

def check_overlap(annotations: pd.DataFrame) -> None:
    """
    Checks for overlapping time intervals of annotated activities in a DataFrame.

    Args:
        annotations (pd.DataFrame): The input DataFrame with 'start' and 'end' columns
        containing datetime objects.

    Raises:
        ValueError: If any activity's 'end' timestamp overlaps with the next activity's 'start' timestamp.
    """
    annotations_number = len(annotations) - 1

    for i in range(annotations_number):
        if annotations.loc[i, 'end'] > annotations.loc[i + 1, 'start']:
            raise ValueError(f"Overlap between {annotations.loc[i, 'end']} {annotations.loc[i, 'annotation']} and {annotations.loc[i + 1, 'start']} {annotations.loc[i + 1, 'annotation']}.")

def parse_annotations(annotations: pd.DataFrame) -> None:
    """
    Checks if all annotations in a DataFrame are within an allowed list of activities.

    Args:
        annotations (pd.DataFrame): The input DataFrame containing an 'annotation' column.

    Raises:
        ValueError: If any annotation is not in the predefined list of activities.
    """
    activities = (
        "airing", "preparing for bed", "sleeping", "getting up", "working out", 
        "preparing breakfast", "eating breakfast", "preparing dinner", "eating dinner", 
        "preparing supper", "eating supper", "preparing a drink", "working", "relaxing", 
        "leaving home", "entering home", "preparing a meal", "eating a meal"
    )

    invalid_annotations = set()
    for annotation in annotations['annotation']:
        if annotation not in activities:
            invalid_annotations.add(annotation)

    if invalid_annotations:
        raise ValueError(f"Invalid annotations found: {invalid_annotations}")

if __name__ == '__main__':
    path_annotation_file = get_annotations_file_path

    annotations = read_csv_to_pandas_dataframe(path_annotation_file)
    parse_timestamps(annotations)
    check_overlap(annotations)
    parse_annotations(annotations)