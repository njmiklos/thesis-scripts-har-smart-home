"""
Categorizes numerical column values in a CSV file as 'low', 'medium', or 'high', and saves the categorized data 
along with annotation labels to a new file.

Environment Configuration:
- Set `INPUTS_DIR` and `OUTPUTS_DIR` in your `.env` file to specify directories for input and output CSV files.
- Input files must include a clearly labeled annotation column, with all other columns containing numerical data.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""

import pandas as pd

from pathlib import Path
from typing import Dict, List, Tuple, Set

from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv, check_if_output_directory_exists
from utils.get_env import get_path_from_env


def get_low_and_high_values(column_data: pd.Series, values_no: int) -> Tuple[Set, Set]:
    """
    Determines the sets of low and high values based on sorted data.

    Args:
        column_data (pd.Series): The column data to be analyzed.
        values_no (int): Number of lowest and highest values to extract.

    Returns:
        Tuple[Set, Set]: A tuple containing the sets of low and high values.
    """
    sorted_values = column_data.sort_values().tolist()

    if values_no < 1:
        raise ValueError(f'Number of lowest/highest values must be at least 1, got {values_no}.')

    if values_no == 1:
        low_values = {sorted_values[0]}
        high_values = {sorted_values[-1]}
    else:
        low_values = set(sorted_values[:values_no])
        high_values = set(sorted_values[-values_no:])
    return low_values, high_values

def categorize_column_values(column_data: pd.Series, values_no: int) -> List[str]:
    """
    Categorizes each value in a column as 'low', 'medium', 'high', or 'none'.

    Args:
        column_data (pd.Series): The column data to categorize.
        values_no (int): Number of lowest and highest values to consider.

    Returns:
        List[str]: List of category labels for the column.
    """
    low_values, high_values = get_low_and_high_values(column_data, values_no)

    categories = []
    for val in column_data:
        if pd.isna(val):
            categories.append('none')
        elif val in low_values:
            categories.append('low')
        elif val in high_values:
            categories.append('high')
        else:
            categories.append('medium')

    return categories

def map_data(df: pd.DataFrame, values_no: int) -> Dict[str, List[str]]:
    """
    Maps numerical columns (data) in the DataFrame to categorical labels.

    Args:
        df (pd.DataFrame): DataFrame with numerical values.
        values_no (int): Number of lowest and highest values to consider for categorization.

    Returns:
        Dict[str, List[str]]: Dictionary mapping each column name to its list of category labels.
    """
    column_mappings = {}
    for column_name in df.columns:
        column_mappings[column_name] = categorize_column_values(df[column_name], values_no)
    return column_mappings

def convert_dict_to_dataframe(dictionary: Dict[str, List[str]], annotation_col_name: str, 
                              annotations_col: pd.Series) -> pd.DataFrame:
    """
    Converts a dictionary of column mappings to a DataFrame and inserts the annotations column at the front.

    Args:
        dictionary (Dict[str, List[str]]:): Dictionary where keys are column names and values are lists of category labels.
        annotation_col_name (str): Name of the annotation column.
        annotations_col (pd.Series): Series representing annotation labels.

    Returns:
        pd.DataFrame: Combined DataFrame with annotations and category labels.
    """
    df = pd.DataFrame(dictionary)
    df.insert(0, annotation_col_name, annotations_col)
    return df

def map_measurements_to_lows_and_highs(file_in_path: Path, file_out_path: Path, annotation_col_name: str, values_no: int) -> None:
    """
    Reads a CSV, categorizes each value in numerical columns as 'low', 'medium', or 'high',
    and saves the resulting DataFrame.

    Args:
        file_in_path (Path): Path to the input CSV file.
        file_out_path (Path): Path to save the categorized output CSV.
        annotation_col_name (str): Name of the annotation column.
        values_no (int): Number of lowest and highest values to categorize. Others become 'medium'.
    """
    df = read_csv_to_dataframe(file_in_path)

    annotations_col = df.pop(annotation_col_name)
    data = df

    mapped_data = map_data(data, values_no)
    output_df = convert_dict_to_dataframe(mapped_data, annotation_col_name, annotations_col)
    save_dataframe_to_csv(output_df, file_out_path)


if __name__ == '__main__':
    values_no = 3
    input_file_name = 'summary_classes_synced_merged_selected.csv'
    output_file_name = 'summary_classes_synced_merged_selected_mapped.csv'
    annotation_col_name = 'annotation'

    input_dir = get_path_from_env('INPUTS_DIR')
    output_dir = get_path_from_env('OUTPUTS_DIR')
    check_if_output_directory_exists(output_dir)

    file_in_path = input_dir / input_file_name
    file_out_path = output_dir / output_file_name

    map_measurements_to_lows_and_highs(file_in_path, file_out_path, annotation_col_name, values_no)