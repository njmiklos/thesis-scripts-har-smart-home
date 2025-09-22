"""
Validates a file listing annotated episodes (i.e., time intervals of annotated activities).
Each row in the file represents one episode, with the following required columns:
- 'start': Start time in milliseconds since epoch (UTC), interpreted in Europe/Berlin timezone.
- 'end': End time in the same format.
- 'annotation': Activity label (e.g., 'sleeping', 'airing').

The validated and optionally gap-annotated file can be used to annotate sensor data or validate predictions.

Refer to `README.md` for full setup and usage instructions.
"""
import pandas as pd

from pathlib import Path
from typing import Set

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv, check_if_output_directory_exists
from data_processing.convert_timestamps import convert_timestamps_from_localized_datetime_to_miliseconds


def parse_timestamps(annotations: pd.DataFrame) -> None:
    """
    Converts 'start' and 'end' columns from epoch milliseconds to datetime.

    Args:
        annotations (pd.DataFrame): DataFrame with 'start' and 'end' columns as integers or strings.
    
    Returns:
        None
    """
    for column in ['start', 'end']:
        try:
            annotations[column] = pd.to_datetime(annotations[column], unit='ms', utc=True).dt.tz_convert('Europe/Berlin')
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format in column '{column}': {e}")

def check_for_overlaps(annotations: pd.DataFrame) -> None:
    """
    Ensures that episodes do not overlap.

    Args:
        annotations (pd.DataFrame): DataFrame with 'start' and 'end' as datetime.

    Returns:
        None
    """
    annotations = annotations.sort_values(by='start').reset_index(drop=True)

    num_annotations = len(annotations) - 1
    for i in range(num_annotations):
        if annotations.loc[i, 'end'] > annotations.loc[i + 1, 'start']:
            raise ValueError(f"Overlap between {annotations.loc[i, 'end']} {annotations.loc[i, 'annotation']} "
                             f"and {annotations.loc[i + 1, 'start']} {annotations.loc[i + 1, 'annotation']}.")

def validate_annotations(annotations: pd.DataFrame, expected_annotations: Set[str]) -> None:
    """
    Validates that all annotations are from a predefined list.

    Args:
        annotations (pd.DataFrame): DataFrame with 'annotation' column.
        expected_annotations (Set[str]): A list of expected annotations.

    Returns:
        None
    """
    invalid = set(annotations['annotation']) - set(expected_annotations)
    if invalid:
        raise ValueError(f"Invalid annotations found: {invalid}")

def annotate_gaps(annotations: pd.DataFrame, annotation_for_gaps: str) -> pd.DataFrame:
    """
    Inserts rows annotated as specified for gaps between episodes longer than 1 second.

    Args:
        annotations (pd.DataFrame): Sorted DataFrame with 'start' and 'end' columns as datetime.
        annotation_for_gaps (str): Label to use for the gaps.

    Returns:
        pd.DataFrame: Annotated DataFrame including gap intervals.
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
                'annotation': annotation_for_gaps
            })

    if gaps:
        gaps_df = pd.DataFrame(gaps)
        annotations = pd.concat([annotations, gaps_df])
    
    annotations = annotations.sort_values(by='start').reset_index(drop=True)

    return annotations

def parse_annotation_file(path_annotation_file: Path, expected_annotations: Set[str], 
                          output_path: Path, annotation_for_gaps: str) -> None:
    """
    Parses and validates an annotation file.

    Args:
        path_annotation_file (Path): Path to the input CSV.
        expected_annotations (Set[str]): A list of expected annotations.
        output_path (Path): Path to save the validated (and possibly extended) CSV.
        annotation_for_gaps (str): Label to annotate gaps. If empty, no gaps are annotated.

    Returns:
        None
    """
    annotations = read_csv_to_dataframe(path_annotation_file)
    parse_timestamps(annotations)
    check_for_overlaps(annotations)
    validate_annotations(annotations, expected_annotations)

    if annotation_for_gaps:
        annotations = annotate_gaps(annotations, annotation_for_gaps)

    annotations = convert_timestamps_from_localized_datetime_to_miliseconds(annotations, 'start')
    annotations = convert_timestamps_from_localized_datetime_to_miliseconds(annotations, 'end')
    save_dataframe_to_csv(annotations, output_path)


if __name__ == '__main__':
    path_annotation_file = get_path_from_env('ANNOTATIONS_FILE_PATH')
    output_dir_path = get_path_from_env('OUTPUTS_PATH')
    output_filename = 'annotations_combined_with_annotated_gaps.csv'
    check_if_output_directory_exists(output_dir_path)
    output_path = get_path_from_env('OUTPUTS_PATH') / output_filename

    annotation_for_gaps = '' # If given empty string, gaps between activities will not be included in the annotation file.
    expected_annotations = {
        "airing", "preparing for bed", "sleeping", "getting up", "working out", 
        "preparing breakfast", "eating breakfast", "preparing dinner", "eating dinner", 
        "preparing supper", "eating supper", "preparing a drink", "working", "relaxing", 
        "leaving home", "entering home", "preparing a meal", "eating a meal", "other"
    }

    parse_annotation_file(path_annotation_file, expected_annotations, output_path, annotation_for_gaps)

