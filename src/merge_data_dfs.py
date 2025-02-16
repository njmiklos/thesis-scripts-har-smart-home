import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional, Tuple

from handle_csv import get_all_csv_files_in_directory, read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from get_env import get_base_path
from summarize_classes import prefix_column_names_with_device_sensor_function
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime, convert_timestamps_from_localized_datetime_to_miliseconds


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
    df = read_csv_to_pandas_dataframe(file_path)

    if skip_unannotated:
        df = df[df['annotation'] != 'other']

    df = prefix_column_names_with_device_sensor_function(df, file_path)
    return df

def merge_synchronized_files_into_single_df(dataset_path: Path, skip_unannotated: bool = False) -> Optional[pd.DataFrame]:
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

def merge_annotated_synchronized_files_into_single_df(dataset_path: Path, skip_unannotated: bool = True) -> Optional[pd.DataFrame]:
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

def subdivide_into_file_per_day(df: pd.DataFrame, output_path: Path):
    """
    Splits the DataFrame into separate files for each day.
    The DataFrame is assumed to have a column 'time' containing
    timestamps in milliseconds since the epoch, UTC Eurpe/Berlin.
    
    Each file contains data for one day (00:00:00 to 23:59:59) in the Europe/Berlin timezone.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        output_path (Path): Directory where the output files will be saved.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')
    
    grouped = df.groupby(df['time'].dt.date)

    for day, group in grouped:
        group = group.copy()    # Avoids SettingWithCopyWarning
        group = convert_timestamps_from_localized_datetime_to_miliseconds(group, 'time')

        filename = output_path / f'data_{day}.csv'
        group.to_csv(filename, index=False)
        print(f'INFO: Saved data for {day} to {filename}.')

def merge_and_save_synchronized_files(dataset_path: Path, output_path: Path, skip_unannotated: bool = True, subdivide_into_days: bool = False) -> None:
    """
    Merges CSV files from a specified dataset directory into a single DataFrame and saves the result to a CSV file.
    The processing includes optional removal of rows with 'annotation' equal to 'other'.

    Args:
        dataset_path (Path): The directory containing the CSV files.
        output_path (Path): The file path where the merged CSV should be saved.
        skip_unannotated (bool, optional): If True, rows with 'annotation' equal to 'other' will be removed
            during file processing. Defaults to True.
    """
    df = merge_annotated_synchronized_files_into_single_df(dataset_path, skip_unannotated)

    if not df.empty:
        if not subdivide_into_days:
            save_pandas_dataframe_to_csv(df, output_path)
        else:
            subdivide_into_file_per_day(df, output_path)
    else:
        print('WARNING: DataFrame empty, skipping.')


if __name__ == '__main__':
    base_path = get_base_path()

    # Adjust before running
    dataset_path = base_path / 'Dataset Transition Activities'

    '''
    skip_unannotated = True
    subdivide_into_days = False

    if subdivide_into_days:
        output_path = dataset_path / 'Synchronized_merged_10s_daily'
    else:
        output_path = dataset_path / 'Synchronized_merged_10s.csv'
    merge_and_save_synchronized_files(dataset_path, output_path, skip_unannotated, subdivide_into_days)
    '''

    # Subdivided only
    dataset_file = dataset_path / 'Transition Activities.csv'
    df = read_csv_to_pandas_dataframe(dataset_file)

    output_path = dataset_path
    subdivide_into_file_per_day(df, output_path)
