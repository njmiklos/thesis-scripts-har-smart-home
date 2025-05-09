import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def get_interquartile_range(data_srs: pd.Series) -> float:
    """
    Calculates the interquartile range (IQR) of a specified column in the DataFrame.

    Args:
        data_srs (pd.Series): The input column of a DataFrame.

    Returns:
        float: The interquartile range of the specified column.

    The IQR is the range between the 25th and 75th percentiles, representing the middle 
    50% of the data. It is a measure of statistical dispersion. A small value means 
    the middle of data has a small spread, i.e., it is tightly packed. Such data can 
    be interpreted as more predictable, stable OR inflexible.

    Example:
    Consider this dataset: 2,4,6,8,10,12,14,16,18,202,4,6,8,10,12,14,16,18,20
    Q1 (25th percentile): The value below which 25% of the data falls, 6.
    Q3 (75th percentile): The value below which 75% of the data falls, 16.
    IQR: 16 - 6 = 10.
    """
    return data_srs.quantile(0.75) - data_srs.quantile(0.25)

def get_iqr_relative_to_whole_value_range(iqr: float, minimum_value: float, maximum_value: float) -> float:
    """
    Calculates the IQR as a percentage relative to the overall data range.

    Args:
        iqr (float): The interquartile range (IQR).
        minimum_value (float): The minimum value in the dataset.
        maximum_value (float): The maximum value in the dataset.

    Returns:
        float: The IQR expressed as a percentage of the total value range.
    """
    values_range = maximum_value - minimum_value
    if values_range == 0:
        return 0
    
    iqr_rel = (iqr / values_range) * 100
    iqr_rel = abs(iqr_rel)

    return iqr_rel

def get_std_relative_to_mean(std: float, mean: float ) -> float:
    """
    Calculates the standard deviation as a percentage relative to the mean.

    Args:
        std (float): The standard deviation of the data.
        mean (float): The mean of the data.

    Returns:
        float: The standard deviation expressed as a percentage of the mean.
    """
    if mean == 0:
        return 0

    std_rel = (std / mean) * 100
    std_rel = abs(std_rel)

    return std_rel

def get_quick_stats_dict(data_srs: pd.Series) -> dict:
    """
    Generates a summary statistics dictionary for a specified column.

    Args:
        data_srs (pd.Series): The input Series containing the data for which statistics are calculated.

    Returns:
        dict: A dictionary containing the following statistics:
            - missing_values_count: Number of missing (NaN) values.
            - infinite_values_count: Number of infinite values.
            - minimum_value: The minimum value in the column.
            - maximum_value: The maximum value in the column.
            - median_value: The median (50th percentile) of the column.
            - mean_value: The mean (average) of the column.
            - standard_deviation: The standard deviation of the column.
            - standard_deviation_relative_to_mean: The standard deviation as a percentage of the mean.
            - iqr: The interquartile range (IQR) of the column.
            - iqr_relative_to_whole_range: The IQR as a percentage of the total range.
    """
    missing_vals = data_srs.isna().sum()
    infs_vals = np.isinf(data_srs).sum()

    # Replaces infinite values with NaN for calculations 
    col_data_clean = data_srs.replace([np.inf, -np.inf], np.nan)
    stats = col_data_clean.describe()

    std_rel = get_std_relative_to_mean(stats['std'], stats['mean'])

    iqr = get_interquartile_range(data_srs)
    iqr_rel = get_iqr_relative_to_whole_value_range(iqr, stats['min'], stats['max'])

    stats_dictionary = {
        'missing values count' : missing_vals,
        'infinite values count' : infs_vals,
        'minimum value' : stats['min'],
        'maximum value' : stats['max'],
        'median value' : stats['50%'],
        'mean value' : stats['mean'],
        'standard deviation' : stats['std'],
        'standard deviation_relative_to_mean' : std_rel,
        'iqr' : iqr,
        'iqr relative to whole range' : iqr_rel
    }

    return stats_dictionary

def calculate_thresholds(df: pd.DataFrame, col: str, mean: float, std_dev: float, num_std_dev: float) -> Tuple[float, float]:
    """
    Calculates the lower and upper thresholds for a specified column in the DataFrame
    based on the mean and standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column for which thresholds are calculated.
        num_std_dev (float): The number of standard deviations to use for threshold calculation.
        If num_std_dev = 1, thresholds cover ~68% of the data (detect standard), 
        num_std_dev = 2 -> ~95%, and num_std_dev = 3 -> ~99.7% (detect outliers).

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper thresholds.
    """
    lower_threshold = mean - num_std_dev * std_dev
    upper_threshold = mean + num_std_dev * std_dev
    return lower_threshold, upper_threshold

def count_readings_out_of_range(df: pd.DataFrame, col: str, 
                                lower_threshold: float, upper_threshold: float) -> Tuple[int, int]:
    """
    Counts the number of readings outside a specified range in a column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to check for out-of-range values.
        lower_threshold (int): The lower threshold value.
        upper_threshold (int): The upper threshold value.

    Returns:
        Tuple[int, int]: A tuple containing the lower and upper thresholds.
    """
    below_lower_threshold = (df[col] < lower_threshold).sum()
    above_upper_threshold = (df[col] > upper_threshold).sum()

    return below_lower_threshold, above_upper_threshold

def get_correlation(merged_df: pd.DataFrame, col1: str, col2: str):
    """
    correlation: Value between -1 and 1. A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation.
    p_value: Determines the statistical significance. Typically, a p-value < 0.05 is considered significant.
    """
    correlation, p_value = pearsonr(merged_df[col1], merged_df[col2])
    print(f"Pearson Correlation Coefficient: {correlation}")
    print(f"P-value: {p_value}")
    return correlation, p_value

def calculate_magnitudes_of_motion_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the magnitude of accelerometer, gyroscope, and magnetometer readings 
    from their respective X, Y, and Z components in a DataFrame. Rounded to 11 decimal positions.

    The input dataframe must contain the following columns:
    ['time', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ']

    Args:
        df (pd.DataFrame): Input dataframe containing sensor readings.

    Returns:
        pd.DataFrame: A dataframe with columns ['time', 'acc', 'gyro', 'mag'],
            where each sensor value represents the calculated magnitude.
    """
    df['acc'] = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)
    df['acc'] = df['acc'].round(11)
    df['gyro'] = np.sqrt(df['gyroX']**2 + df['gyroY']**2 + df['gyroZ']**2)
    df['gyro'] = df['gyro'].round(11)
    df['mag'] = np.sqrt(df['magX']**2 + df['magY']**2 + df['magZ']**2)
    df['mag'] = df['mag'].round(11)

    result_df = df[['time', 'acc', 'gyro', 'mag']]
    return result_df

def get_root_mean_square_error(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> float:
    """
    Calculates Root Mean Square Error (RMSE) between two DataFrames to quantify 
    the differences between them and to check how well (e.g.,) the resampling process 
    preserves the original structure.

    Args:
        df1 (pd.DataFrame): The first DataFrame (e.g., original data).
        df2 (pd.DataFrame): The second DataFrame (e.g., resampled data).
        column (str): The column name for which RMSE should be calculated.

    Returns:
        float: The computed RMSE value, rounded to 5 decimal places.
    
    Notes:
        - Low RMSE: Indicates that the df2 data closely follows df1 values.
        - High RMSE: Suggests that there are significant deviations introduced in df2 from df1.
        - RMSE Units: Same as the unit of the column being evaluated.
    """
    if column not in df1.columns or column not in df2.columns:
        raise ValueError(f"Column '{column}' not found in both DataFrames.")

    # If the DataFrames have different lengths, the longer one is  trimmed to the minimum length for comparison
    min_length = min(len(df1), len(df2))
    df1_aligned = df1.iloc[:min_length].reset_index(drop=True)
    df2_aligned = df2.iloc[:min_length].reset_index(drop=True)

    # Calculates distances between the values
    rmse = mean_squared_error(df1_aligned[column], df2_aligned[column]) ** 0.5
    return round(rmse, 5)

def get_root_mean_square_error_srs(data_srs1: pd.Series, data_srs2: pd.Series) -> float:
    """
    Calculates Root Mean Square Error (RMSE) between two columns of two DataFrames to quantify 
    the differences between them and to check how well the resampling process 
    preserves the original structure.

    Args:
        data_srs1 (pd.DataFrame): The first column (e.g., original data).
        data_srs2 (pd.DataFrame): The second column (e.g., resampled data).

    Returns:
        float: The computed RMSE value, rounded to 5 decimal places.
    
    Notes:
        - Low RMSE: Indicates that the df2 data closely follows df1 values.
        - High RMSE: Suggests that there are significant deviations introduced in df2 from df1.
        - RMSE Units: Same as the unit of the column being evaluated.
    """
    # If the DataFrames have different lengths, the longer one is  trimmed to the minimum length for comparison
    min_length = min(len(data_srs1), len(data_srs2))
    data_srs1_aligned = data_srs1.iloc[:min_length].reset_index(drop=True)
    data_srs2_aligned = data_srs2.iloc[:min_length].reset_index(drop=True)

    # Calculates distances between the values
    rmse = mean_squared_error(data_srs1_aligned, data_srs2_aligned) ** 0.5
    return round(rmse, 5)