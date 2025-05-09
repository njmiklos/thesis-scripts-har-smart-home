"""
This files validates a file listing all annotated episodes. 

In the file, every episode is a row. It must include the following columns:
    - `start`: Episode start time (e.g., 1733353200000 – UTC milliseconds, Europe/Berlin timezone).
    - `end`: Episode end time (see the format above).
    - `annotation`: Activity class (e.g., 'sleeping', 'airing').
"""
import pandas as pd

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv


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

def annotate_gaps(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates as 'other' intervals between annotated activities in the DataFrame that are longer than 00:00:01.

    Args:
        annotations (pd.DataFrame): The input DataFrame with 'start' and 'end' columns
        containing timestamps.

    Returns:
        pd.DataFrame: A new DataFrame including the 'other' intervals.
    """
    gaps = []

    for i in range(len(annotations) - 1):
        current_end = int(annotations.loc[i, 'end'])
        next_start = int(annotations.loc[i + 1, 'start'])

        if next_start - current_end > 1000:
            gap_between_episodes = True
        else:
            gap_between_episodes = False

        if gap_between_episodes:
            gaps.append({'start': current_end + 1000,
                'end': next_start - 1000,
                'annotation': 'other'
            })

    if gaps:
        gaps_df = pd.DataFrame(gaps)
        annotations = pd.concat([annotations, gaps_df])
    
    annotations = annotations.sort_values(by='start').reset_index(drop=True)

    return annotations


if __name__ == '__main__':
    path_annotation_file = get_path_from_env('ANNOTATIONS_FILE_PATH')

    annotations = read_csv_to_dataframe(path_annotation_file)
    parse_timestamps(annotations)
    check_overlap(annotations)
    parse_annotations(annotations)

    annotations_with_annotated_gaps = annotate_gaps(annotations)
    file_out = 'annotations_combined_with_annotated_gaps.csv'
    path_out =  get_path_from_env('OUTPUTS_PATH') / file_out
    save_dataframe_to_csv(annotations_with_annotated_gaps, path_out)

