import pandas as pd

from typing import List
from pathlib import Path

from data_processing.convert_timestamps import (convert_timestamps_from_miliseconds_to_localized_datetime_srs, 
                                                convert_timestamps_from_localized_datetime_to_miliseconds_srs)
from data_analysis.summarize.summarize_episodes_counts_and_lenghts import generate_summary_dataframe
from data_analysis.summarize.summarize_classes import process_file


def infer_data_collection_days_from_time_column(srs: pd.Series) -> List[str]:
    """
    Infers and returns a list of unique data collection dates from a timestamp column.

    Args:
        srs (pd.Series): A Pandas Series containing timestamp values.

    Returns:
        List[str]: A sorted list of unique dates in 'YYYY-MM-DD' format.
    """
    srs = convert_timestamps_from_miliseconds_to_localized_datetime_srs(srs)
    days = pd.to_datetime(srs).dt.date.unique()
    days = [day.strftime('%Y-%m-%d') for day in sorted(days)]
    srs = convert_timestamps_from_localized_datetime_to_miliseconds_srs(srs)
    return days

def infer_classes_from_class_column(srs: pd.Series) -> List[str]:
    """
    Infers and returns a sorted list of unique class values from a class column.

    Args:
        srs (pd.Series): A Pandas Series containing class values.

    Returns:
        List[str]: A sorted list of unique class values as strings.
    """
    classes = pd.to_datetime(srs).dt.date.unique()
    classes = [str(cls) for cls in sorted(classes)]
    return classes

def infer_episodes_counts_and_lenghts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a combined summary dataframe with statistics for each annotation class 
    and overall statistics.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        pd.DataFrame: Combined summary dataframe.
    """
    summary_df = generate_summary_dataframe(df)
    return summary_df

def infer_class_measurements(file_path: Path) -> pd.DataFrame:
    """
    Processes a dataset CSV file to provide a summary of all classes as a DataFrame.
    Every rows is a class, and columns show a mean and a standard deviation of every
    measurement in the class.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The processed dataframe with aggregated values and updated column names.
    """
    df = process_file(file_path)
    return df