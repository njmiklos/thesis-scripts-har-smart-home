import pandas as pd
from typing import List

from utils.get_env import get_annotations_file_path, get_output_path
from utils.handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv


def convert_milliseconds_to_minutes(milliseconds: int) -> float:
    """
    Convert time from milliseconds to minutes.

    Args:
        milliseconds (int): Time in milliseconds.

    Returns:
        float: Time in minutes.
    """
    return milliseconds / (1000 * 60)

def calculate_total_recording_duration(df: pd.DataFrame) -> int:
    """
    Calculate the total recording time in milliseconds 
    from the earliest start to the latest end timestamp.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        int: The total recording time in milliseconds.
    """
    recording_start = df['start'].iloc[0]
    recording_end = df['end'].iloc[-1]
    return recording_end - recording_start

def calculate_annotated_percentage(total_recording_ms: int, total_annotated_ms: int) -> float:
    """
    Calculate the percentage of annotated time relative to total recording time.

    Args:
        total_recording_ms (int): Total recording time in milliseconds.
        total_annotated_ms (int): Total annotated time in milliseconds.

    Returns:
        float: Percentage of annotated time.
    """
    return (total_annotated_ms * 100) / total_recording_ms

def calculate_total_annotated_duration(df: pd.DataFrame) -> int:
    """
    Calculate the total annotated time in milliseconds.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        int: Total annotated time in milliseconds.
    """
    durations = df['end'] - df['start']
    return durations.sum()

def get_unique_annotations(df: pd.DataFrame) -> List[str]:
    """
    Retrieve unique annotation classes from the dataframe.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        List[str]: List of unique annotation classes.
    """
    return df['annotation'].unique()

def calculate_annotation_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate occurrence count, shortest, longest, average, and total 
    durations for a specific annotation class.

    Args:
        df (pd.DataFrame): The dataframe for a single annotation class.

    Returns:
        dict: Dictionary containing annotation statistics.
    """
    episode_count = len(df)
    durations = df['end'] - df['start']
    
    return {
        'count': episode_count,
        'min': convert_milliseconds_to_minutes(durations.min()),
        'max': convert_milliseconds_to_minutes(durations.max()),
        'mean': convert_milliseconds_to_minutes(durations.mean()),
        'total': convert_milliseconds_to_minutes(durations.sum())
    }

def generate_annotation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of statistics for each annotation class.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Summary dataframe with statistics for each annotation class.
    """
    annotation_classes = get_unique_annotations(df)
    summary_columns = ['class', 'count', 'min', 'max', 'mean', 'total']
    annotation_summary_df = pd.DataFrame(columns=summary_columns)

    for annotation_class in annotation_classes:
        class_rows = df[df['annotation'] == annotation_class]
        class_statistics = calculate_annotation_statistics(class_rows)
        class_statistics['class'] = annotation_class

        annotation_summary_df = pd.concat(
            [annotation_summary_df, pd.DataFrame([class_statistics])],
            ignore_index=True
        )

    return annotation_summary_df

def generate_total_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an overall summary of all annotations combined.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Summary dataframe with total statistics.
    """
    total_columns = ['class', 'count', 'min', 'max', 'mean', 'total_recorded', 'total_annotated', 
                     'annotated_percent']
    total_summary_df = pd.DataFrame(columns=total_columns)

    durations = df['end'] - df['start']

    total_recorded_ms = calculate_total_recording_duration(df)
    total_annotated_ms = calculate_total_annotated_duration(df)

    total_statistics = {
        'class': 'TOTAL',
        'count': len(df),
        'min': convert_milliseconds_to_minutes(durations.min()),
        'max': convert_milliseconds_to_minutes(durations.max()),
        'mean': convert_milliseconds_to_minutes(durations.mean()),
        'total_recorded': convert_milliseconds_to_minutes(total_recorded_ms),
        'total_annotated': convert_milliseconds_to_minutes(total_annotated_ms),
        'annotated_percent': calculate_annotated_percentage(total_recorded_ms, total_annotated_ms)
    }

    total_summary_df = pd.concat([total_summary_df, pd.DataFrame([total_statistics])], ignore_index=True)

    return total_summary_df

def round_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round all time columns to two decimal places.

    Args:
        df (pd.DataFrame): The summary dataframe.

    Returns:
        pd.DataFrame: Summary dataframe with rounded time values.
    """
    time_columns = [col for col in df.columns if col not in ['class', 'count']]
    df[time_columns] = df[time_columns].round(2)
    return df

def generate_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a combined summary dataframe with statistics for each annotation class 
    and overall statistics.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Combined summary dataframe.
    """
    annotation_summary_df = generate_annotation_summary(df)
    total_summary_df = generate_total_summary(df)
    summary_df = pd.concat([annotation_summary_df, total_summary_df], ignore_index=True)
    summary_df = round_time_columns(summary_df)

    return summary_df

if __name__ == '__main__':
    # Get file paths
    annotation_file_path = get_annotations_file_path()
    output_file_name = 'summary_episode_times_and_occurances.csv'
    output_file_path = get_output_path() / output_file_name

    # Read input annotations
    annotations_df = read_csv_to_pandas_dataframe(annotation_file_path)

    # Generate and save summary
    summary_df = generate_summary_dataframe(annotations_df)
    save_pandas_dataframe_to_csv(summary_df, output_file_path)

    print(f"Summary saved to {output_file_path}")
