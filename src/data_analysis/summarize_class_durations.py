"""
This script analyzes the annotation file to compute time-based statistics 
for each annotation class and for the dataset as a whole.

It calculates:
- Episode counts per class
- Minimum, maximum, mean, and total duration (in minutes) for each class
- Total recorded time, total annotated time, and percent of all annotated data

Example output:
```
class,count,min,max,mean,total,total_recorded,total_annotated,total_percent
sleeping,14,66.97,514.98,433.13,6063.87,,,
getting up,13,13.73,28.65,19.16,249.05,,,
preparing breakfast,13,13.33,37.23,24.48,318.2,,,
```

Environment Configuration:
- Set `ANNOTATIONS_FILE_PATH` and `OUTPUTS_PATH` in your `.env` file.
- Input CSV must contain 'start', 'end', and 'annotation' columns.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
import pandas as pd

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv
from data_analysis.summarize_sensor_stats_by_class import extract_unique_classes


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

def calculate_annotated_time_percentage(total_recording_ms: int, total_annotated_ms: int) -> float:
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

def generate_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of statistics for each annotation class.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Summary dataframe with statistics for each annotation class.
    """
    annotation_classes = extract_unique_classes(df)
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
        'annotated_percent': calculate_annotated_time_percentage(total_recorded_ms, total_annotated_ms)
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

def generate_and_save_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a combined summary dataframe with statistics for each annotation class 
    and overall statistics.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Combined summary dataframe.
    """
    annotation_summary_df = generate_class_summary(df)
    total_summary_df = generate_total_summary(df)
    summary_df = pd.concat([annotation_summary_df, total_summary_df], ignore_index=True)
    summary_df = round_time_columns(summary_df)

    save_dataframe_to_csv(summary_df, output_file_path)

    print(f'Summary saved to {output_file_path}')


if __name__ == '__main__':
    annotation_file_path = get_path_from_env('ANNOTATIONS_FILE_PATH')
    output_file_name = 'summary_episode_times_and_occurances.csv'
    output_file_path = get_path_from_env('OUTPUTS_PATH') / output_file_name

    annotations_df = read_csv_to_dataframe(annotation_file_path)
    summary_df = generate_and_save_summary_dataframe(annotations_df)
