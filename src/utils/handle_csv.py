import pandas as pd
from pathlib import Path
from typing import List


def read_csv_to_pandas_dataframe(path_file: Path) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        path_file (Path): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(path_file)
    return df

def save_pandas_dataframe_to_csv(df: pd.DataFrame, path_file: Path) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path_file (Path): The file path where the DataFrame should be saved.

    Returns:
        None
    """
    df.to_csv(path_file, index=False)
    return None

def get_csv_columns(file_path: Path) -> List[str]:
    """
    Reads only the column names from a CSV file.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        List[str]: List of column names in the CSV file.
    """
    with open(file_path, 'r') as f:
        # Read only the first line to get the column names
        first_line = f.readline().strip()
    column_names = first_line.split(',')
    return column_names

def infer_frequency(df: pd.DataFrame, time_col: str) -> str:
    """
    Infers the most common frequency in the time column of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the time column.

    Returns:
        str: Pandas-compatible frequency string.
    """
    inferred_freq = pd.infer_freq(df[time_col])
    if inferred_freq is None:
        raise ValueError("Could not infer frequency. Ensure timestamps are evenly spaced.")
    return inferred_freq

def get_all_csv_files_in_directory(dir_path: Path):
    files = sorted(dir_path.glob('*.csv'))
    return files


