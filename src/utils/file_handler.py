import pandas as pd
import json

from pathlib import Path
from typing import List


def read_csv_to_dataframe(path_file: Path) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        path_file (Path): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(path_file)
    return df

def save_dataframe_to_csv(df: pd.DataFrame, path_file: Path) -> None:
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

def get_all_csv_files_in_directory(dir_path: Path) -> List[Path]:
    """
    Retrieves and returns all .csv files in the specified directory, sorted alphabetically.

    Args:
        dir_path (Path): Path to the directory to search.

    Returns:
        List[Path]: A list of Path objects pointing to .csv files in the directory.
    """
    files = sorted(dir_path.glob('*.csv'))
    return files

def get_all_json_files_in_directory(dir_path: Path) -> List[Path]:
    """
    Retrieves and returns all .json files in the specified directory, sorted alphabetically.

    Args:
        dir_path (Path): Path to the directory to search.

    Returns:
        List[Path]: A list of Path objects pointing to .json files in the directory.
    """
    files = sorted(dir_path.glob('*.json'))
    return files

def load_json_file(file_path: Path) -> dict:
    """
    Loads a JSON file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as file:    
        return json.load(file)

def save_to_json_file(output_dir_path: Path, dictionary: dict, output_file_name: str = 'classification_report.json'):
    """
    Saves the given dictionary to a JSON file.

    Args:
        output_dir_path (Path): Path to the directory where report files are stored.
        dictionary (dict): The dictionary to be saved.
        output_file_name (Optional[str]): Filename for the report, defaults to 'classification_report.json'.
    """
    check_if_output_directory_exists(output_dir_path)
    output_path = output_dir_path / output_file_name

    with open(output_path, 'w') as f:
        json.dump(dictionary, f, indent=4)
    
    print(f'Saved {output_path}.')

def check_if_output_directory_exists(directory: Path) -> None:
    """
    Makes sure the specified directory exists, creates it if it does not.

    directory (Path): The directory path to be checked.

    Returns:
        None
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f'Required directory was missing, created {directory}.')