import logging
import numpy as np
import pandas as pd

from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime
from explore_data_pandas import count_readings_out_of_range, calculate_thresholds, get_interquartile_range, get_iqr_relative_to_whole_value_range
from infer_sensor_metadata import infer_expected_sampling_rate
from get_env import get_base_path
from get_logger import get_logger
from handle_csv import read_csv_to_pandas_dataframe, get_all_csv_files_in_directory


def report_iqr_values(df: pd.DataFrame, col: str) -> str:
    """
    Returns the interquartile range (IQR) and its relative proportion of the full range
    for a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column for which the IQR is printed.

    Returns:
        str: A summary of results.

    The function calculates the IQR using the `get_interquartile_range` function and
    determines its percentage relative to the full range of the data (max - min). 
    This helps to understand the spread of the middle 50% of the data in the context
    of the column's overall range.
    """
    iqr = get_interquartile_range(df, col)
    max_value = df[col].max()
    min_value = df[col].min()
    iqr_relative = get_iqr_relative_to_whole_value_range(iqr, min_value, max_value)
    report = str(f'IQR: {iqr:.2f}, {iqr_relative:.2f}% of the whole range') # 50% of values are x% of the range between min and max values
    return report

def report_stats(df: pd.DataFrame, col: str) -> str:
    report = []
    num_missing = df[col].isna().sum()
    num_infs = np.isinf(df[col]).sum()
    
    # Replaces infinite values with NaN for calculations 
    # (works on a copy to avoid modifying original data)
    col_data = df[col].replace([np.inf, -np.inf], np.nan)
    
    stats = col_data.describe()

    report.append(f'Missing values: {num_missing}')
    report.append(f'Infinite values: {num_infs}')
    report.append(f'Minimum: {stats["min"]:.2f}')
    report.append(f'Maximum: {stats["max"]:.2f}')
    report.append(f'Median: {stats["50%"]:.2f}')
    report.append(f'Mean: {stats["mean"]:.2f}')
    std_dev_relative = (stats['std'] / stats['mean']) * 100
    std_dev_relative = abs(std_dev_relative)
    report.append(f'Standard deviation: {stats["std"]:.2f}, {std_dev_relative:.2f}% of the mean')
    report.append(report_iqr_values(df, col))

    return '\n'.join(report)

def report_outliers(df: pd.DataFrame, col: str, lower_threshold: float, upper_threshold: float) -> str:
    """
    Prints outliers below and above specified thresholds in a DataFrame column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to check for outliers.
        lower_threshold (float): The lower threshold value.
        upper_threshold (float): The upper threshold value.

    Returns:
        str: A summary of outliers.
    """
    report = []
    values_below = df[df[col] < lower_threshold]
    values_below = values_below.copy()
    values_above = df[df[col] > upper_threshold]
    values_above = values_above.copy()

    if (not values_below.empty) or (not values_above.empty):
        report.append('\nOutliers (num_std_dev=3)')
    else:
        report.append('\nNo outliers (num_std_dev=3)')

    if not values_below.empty:
        values_below = convert_timestamps_from_miliseconds_to_localized_datetime(values_below, 'time')
        report.append(f'10 smallest values below lower threshold:')
        report.append(values_below.nsmallest(10, col).to_string(index=False))

    if not values_above.empty:
        values_above = convert_timestamps_from_miliseconds_to_localized_datetime(values_above, 'time')
        report.append(f'10 largest values above upper threshold:')
        report.append(values_above.nlargest(10, col).to_string(index=False))
    
    return '\n'.join(report)

def report_thresholds(df: pd.DataFrame, col: str, std_dev: float) -> str:
    """
    Checks thresholds for 1, 2, and 3 standard deviations and counts readings
    outside these thresholds. Returns the formatted report string.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column to check thresholds for.
        std_dev (float): The standard deviation of the column.

    Returns:
        str: Formatted threshold report.
    """
    report = []
    mean_value = df[col].mean()

    for i in range(1, 4):
        num_std_dev = i
        lower_threshold, upper_threshold = calculate_thresholds(df, col, mean_value, std_dev, num_std_dev)
        report.append(f'Thresholds above {num_std_dev} standard deviations (num_std_dev={num_std_dev})')
        report.append(f'\t- Lower threshold: {lower_threshold}')
        report.append(f'\t- Upper threshold: {upper_threshold}')

        below_lower_threshold, above_upper_threshold = count_readings_out_of_range(df, col, lower_threshold, upper_threshold)
        total_outliers = below_lower_threshold + above_upper_threshold
        portion = (total_outliers / len(df[col])) * 100
        report.append(f'No. of readings below lower threshold: {below_lower_threshold}')
        report.append(f'No. of readings above upperthreshold: {above_upper_threshold}')
        report.append(f'This is {portion:.2f}% of the data outside of the thresholds.')

        if i == 3:  # When num_std_dev = 3, thresholds cover ~99.7% (outliers)
            outliers = report_outliers(df, col, lower_threshold, upper_threshold)
            report.append(outliers)

    return '\n'.join(report)

def report_sampling_rates(df: pd.DataFrame, col: str, expected_rate: float) -> str:
    """
    Analyzes the sampling rate and provides statistical details for the given dataframe and column.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data to analyze.
        col (str): Column name to analyze.
        expected_rate (float): Expected sampling rate in seconds.
    
    Returns:
        str: Formatted threshold report.
    """
    report = []

    df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')

    # Calculate time differences
    df['time_diff'] = df['time'].diff().dt.total_seconds()

    # Average sampling rate
    avg_sampling_rate = df['time_diff'].mean()
    report.append(f'Average Sampling Rate: {avg_sampling_rate:.2f} seconds')

    # Median sampling rate
    median_sampling_rate = df['time_diff'].median()
    report.append(f'Median Sampling Rate: {median_sampling_rate:.2f} seconds')

    # Check for irregular intervals (greater than 1.5x expected rate)
    irregular_intervals = df[df['time_diff'] > expected_rate * 1.5]
    report.append(f'No. of irregular intervals: {len(irregular_intervals)}')

    if not irregular_intervals.empty:
        report.append(f'Irregular intervals (10 largest samples):')
        irregular_samples = irregular_intervals[['time', 'time_diff']].nlargest(10, 'time_diff')
        report.append(irregular_samples.to_string(index=False))

    return '\n'.join(report)

def process_file(logger: logging.Logger) -> None:
    """
    Processes a CSV file by generating statistical reports and visualizations.

    This function reads a CSV file, performs statistical analysis, and generates various plots 
    (histogram, scatter, timeseries plots) for each column except the 'time' column.
    The generated plots and reports are saved to the specified output directory.

    Args:
        logger (logging.Logger): Logger instance to log messages and warnings during processing.
        plots_output_path (Path): Directory path where the generated plots will be saved.
        file_path (Path): Path to the input CSV file to be processed.
        timeseries (bool, optional): Whether to generate time series plots. Defaults to True.
        scatter (bool, optional): Whether to generate scatter plots. Defaults to True.
        histogram (bool, optional): Whether to generate histogram plots. Defaults to True.
        sample_size (float, optional): Proportion of the dataset to be used for scatter plots.
            For example, `0.5` means 50% of the data will be sampled for visualization. 
            A smaller value (e.g., `0.1`) is recommended for large datasets.

    Returns:
        None
    """
    report = []
    df = read_csv_to_pandas_dataframe(file_path)

    if not df.empty:
        columns = df.columns
        for col in columns:
                if col != 'time':
                        device_and_sensor = file_path.stem
                        report.append(f'\n=== {device_and_sensor} {col} ===')

                        report.append(f'\n--- Statistics ---')
                        quick_stats = report_stats(df, col)
                        report.append(quick_stats)

                        report.append(f'\n--- Thresholds ---')
                        thresholds = report_thresholds(df, col, df[col].std())
                        report.append(thresholds)

                        report.append(f'\n--- Sampling Rates ---')
                        expected_sampling_rate_in_seconds = infer_expected_sampling_rate(col)

                        if expected_sampling_rate_in_seconds >= 0:
                            report.append(f'INFO: Setting sampling rate for {col} to {expected_sampling_rate_in_seconds}.')

                            sampling_rates = report_sampling_rates(df, col, expected_sampling_rate_in_seconds)
                            report.append(sampling_rates)
                        else:
                            report.append(f'WARNING: No sampling rate specified for {col}, skipping.')
                            print(f'WARNING: No sampling rate specified for {col}, skipping.')

                        logger.info('\n'.join(map(str, report)))
                        report.clear()

    else:
        print(f'WARNING: Data is empty for {file_path.name}. Skipping.')
        logger.warning(f'Data is empty for {file_path.name}. Skipping.')


if __name__ == '__main__':
    base_path = get_base_path()

    # Set before running
    log_filename = 'explore_data_pandas_singles_raw'
    dataset_path = base_path / 'Raw'
    sample_size = 1.0   # 100%

    logger = get_logger(log_filename)

    file_paths = get_all_csv_files_in_directory(dataset_path)
    no_files = len(file_paths)
    if no_files > 0:
        counter = 1
        for file_path in file_paths:
            logger.info(f'INFO: Processing file {counter}/{no_files} {file_path.name}.')
            print(f'Processing file {counter}/{no_files} {file_path.name}...')

            if 'motion' in str(file_path.stem):
                sample_size = 0.1
            else:
                sample_size = 0.5

            process_file(logger=logger, file_path=file_path, sample_size=sample_size)

            logger.info(f'INFO: Processed file {counter}/{no_files} {file_path.name}')
            counter = counter + 1

        logger.info(f'INFO: Processed all files.')
        print(f'Processed all files.')
    else:
        logger.warning(f'No files found in the specified directory.')