"""
This script merges time-series sensor data from annotated CSV files and performs data cleaning
(missing/infinite value handling) before saving the result to a single output file.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file.
- Refer to `README.md` for full setup and usage instructions.
"""
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional

from utils.file_handler import (get_all_csv_files_in_directory, read_csv_to_dataframe, 
                                check_if_output_directory_exists, save_dataframe_to_csv)
from utils.get_env import get_path_from_env
from data_analysis.summarize_sensor_stats_by_class import prefix_column_names_with_sensor_metadata


def process_file(file_path: Path, skip_unannotated: bool = True) -> pd.DataFrame:
    """
    Reads and processes a CSV file by updating its column names with device and sensor information,
    and optionally removing rows with an 'annotation' value of 'other'.

    Args:
        file_path (Path): The path to the CSV file.
        skip_unannotated (bool, optional): If True, rows where the 'annotation' column equals 'other'
            will be dropped. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with updated column names and (optionally) filtered rows.
    """
    df = read_csv_to_dataframe(file_path)

    if skip_unannotated:
        df = df[df['annotation'] != 'other']

    df = prefix_column_names_with_sensor_metadata(df, file_path)
    return df

def merge_files_on_time(dataset_path: Path, skip_unannotated: bool = False) -> Optional[pd.DataFrame]:
    """
    Merges multiple CSV files from a specified directory into a single DataFrame.
    Each file is processed (including optional removal of rows with 'annotation' equal to 'other'),
    and the resulting DataFrames are merged on the 'time' and 'annotation' columns using an outer join.

    Args:
        dataset_path (Path): The directory containing the CSV files.

    Returns:
        Optional[pd.DataFrame]: The merged DataFrame if files were found and processed successfully;
            otherwise, None.
    """
    files = get_all_csv_files_in_directory(dataset_path)

    if files:
        no_files = len(files)
        counter = 1

        final_df = None
        for file in files:
            print(f'INFO: Processing file {counter}/{no_files} {file.name}...')
            
            sub_df = process_file(file, skip_unannotated)
            if final_df is not None:
                final_df = pd.merge(final_df, sub_df, on=['time'], how='outer')
            else:
                final_df = sub_df

            counter = counter + 1
        
        if not final_df.empty:
            cleaned_df = clean(final_df)

            print('INFO: Merging complete.')

            print(cleaned_df.head())
            return cleaned_df
        else:
            print('WARNING: Final DataFrame is empty.')
            return None
        
    else:
        print('WARNING: No CSV files found. Exiting.')
        return None

def merge_files_on_time_and_annotation(dataset_path: Path, skip_unannotated: bool = True) -> Optional[pd.DataFrame]:
    """
    Merges multiple CSV files from a specified directory into a single DataFrame.
    Each file is processed (including optional removal of rows with 'annotation' equal to 'other'),
    and the resulting DataFrames are merged on the 'time' and 'annotation' columns using an outer join.

    Args:
        dataset_path (Path): The directory containing the CSV files.
        skip_unannotated (bool, optional): If True, each CSV file will be processed to drop rows where
            'annotation' is 'other'. Defaults to True.

    Returns:
        Optional[pd.DataFrame]: The merged DataFrame if files were found and processed successfully;
            otherwise, None.
    """
    files = get_all_csv_files_in_directory(dataset_path)

    if files:
        no_files = len(files)
        counter = 1

        final_df = None
        for file in files:
            print(f'INFO: Processing file {counter}/{no_files} {file.name}...')
            
            sub_df = process_file(file, skip_unannotated)
            if final_df is not None:
                final_df = pd.merge(final_df, sub_df, on=['time', 'annotation'], how='outer')
            else:
                final_df = sub_df

            counter = counter + 1
        
        if not final_df.empty:
            cleaned_df = clean(final_df)

            print('INFO: Merging complete.')

            print(cleaned_df.head())
            return cleaned_df
        else:
            print('WARNING: Final DataFrame is empty.')
            return None
        
    else:
        print('WARNING: No CSV files found. Exiting.')
        return None

def report_missing_and_infinite_values(df: pd.DataFrame, numeric_cols: pd.Index) -> None:
    """
    Reports the number of missing and infinite values for each numeric column.

    Parameters
        df : pd.DataFrame
            The DataFrame to analyze.
        numeric_cols : pd.Index
            A list of numeric columns to check.

    Returns
        None
    """
    for col in numeric_cols:
        num_missing = df[col].isna().sum()
        num_infs = np.isinf(df[col]).sum()
        print(f'Column: {col} - Missing values: {num_missing}, Infinite values: {num_infs}')

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values by applying different strategies based on column names.
    - For columns containing 'magnitude', missing values are filled with 0.
    - For all other columns, missing values are backfilled first, then forward-filled.

    Parameters
        df : pd.DataFrame
            The DataFrame to process.

    Returns
        pd.DataFrame
            The DataFrame with missing values handled.
    """
    magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower()]
    for col in magnitude_cols:
        df[col] = df[col].fillna(0)

    non_magnitude_cols = [col for col in df.columns if col not in magnitude_cols]
    df[non_magnitude_cols] = df[non_magnitude_cols].bfill().ffill()

    return df

def round_infinite_values(df: pd.DataFrame, numeric_cols: pd.Index) -> pd.DataFrame:
    """
    Rounds infinite values to 13 decimal places in numeric columns.

    Parameters
        df : pd.DataFrame
            The DataFrame to process.
        numeric_cols : pd.Index
            A list of numeric columns to check.

    Returns
        pd.DataFrame
            The DataFrame with infinite values rounded.
    """
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: round(x, 13) if np.isinf(x) else x)  
    
    return df

def clean(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Cleans a pandas DataFrame by reporting numeric column issues, handling missing data,
    and adjusting numeric values with infinite entries.

    Parameters
    df : pd.DataFrame
        The DataFrame to be cleaned.

    Returns
    pd.DataFrame
        A new DataFrame with rows containing missing values removed and numeric columns updated
        by rounding infinite values to 13 decimal places.
    """
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        report_missing_and_infinite_values(df, numeric_cols)
        df = fill_missing_values(df)
        df = round_infinite_values(df, numeric_cols)

        return df
    else:
        print('WARNING: DataFrame empty, skipping.')

def merge_and_save_files(dataset_path: Path, output_path: Path, skip_unannotated: bool = True) -> None:
    """
    Merges CSV files from a specified dataset directory into a single DataFrame and saves the result to a CSV file.
    The processing includes optional removal of rows with 'annotation' equal to 'other'.

    Args:
        dataset_path (Path): The directory containing the CSV files.
        output_path (Path): The file path where the merged CSV should be saved.
        skip_unannotated (bool, optional): If True, rows with 'annotation' equal to 'other' will be removed
            during file processing. Defaults to True.
    """
    df = merge_files_on_time_and_annotation(dataset_path, skip_unannotated)
    save_dataframe_to_csv(df, output_path)


if __name__ == '__main__':
    dataset_filename = 'Transition Activities.csv'
    skip_unannotated = True

    input_dir = get_path_from_env('INPUTS_PATH')
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir)

    merge_and_save_files(input_dir, output_dir, skip_unannotated)
