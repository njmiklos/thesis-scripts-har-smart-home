"""
This module provides utilities for converting between different timestamp formats 
(ISO8601, Unix milliseconds, nanoseconds) and timezones (UTC and Europe/Berlin). 
It supports both `DataFrame` and `Series` transformations to accommodate flexible 
time-series processing pipelines.

Timezone Handling Notes:
- `tz_localize` is used when working with timestamps that are stored without timezone information 
  (e.g., Unix milliseconds or naive strings).
- `tz_convert` is used when the timestamps already have timezone information but need to be shifted 
  to another timezone (e.g., UTC to Europe/Berlin or vice versa).
"""
import pandas as pd


def convert_timestamps_from_iso8601_to_localized_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Converts the specified time column from ISO8601 formatted string to 
    timezone-aware datetime in Europe/Berlin timezone.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time column in ISO8601 format.
        time_col (str): Name of the time column to convert.

    Returns:
        pd.DataFrame: Modified DataFrame with the time column as localized datetime,
        e.g., 2000-01-01 01:00:00+01:00.
    """
    df[time_col] = pd.to_datetime(df[time_col], format='ISO8601', utc=True)  # Parse ISO8601 to UTC datetime
    df[time_col] = df[time_col].dt.tz_convert('Europe/Berlin')  # Convert to Berlin timezone
    return df
    
def convert_timestamps_from_nanoseconds_to_milliseconds(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Converts the specified time column in a DataFrame from nanoseconds (InfluxDB default) 
    to milliseconds since Unix epoch in UTC, adjusted for Europe/Berlin timezone.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time column in nanoseconds.
        time_col (str): Name of the time column to convert.

    Returns:
        pd.DataFrame: Modified DataFrame with the time column in milliseconds since Unix epoch.
    """
    df[time_col] = pd.to_datetime(df[time_col], utc=True)  # Converts to datetime in UTC
    df[time_col] = df[time_col].dt.tz_convert('Europe/Berlin')  # Adjusts to Berlin timezone
    df[time_col] = df[time_col].view('int64') // 10**6  # Converts to milliseconds since Unix epoch
    return df

def convert_timestamps_from_miliseconds_to_localized_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Converts the specified time column from milliseconds since Unix epoch to 
    timezone-aware datetime in Europe/Berlin timezone.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time column in milliseconds.
        time_col (str): Name of the time column to convert.

    Returns:
        pd.DataFrame: Modified DataFrame with the time column as localized datetime,
        e.g., 2000-01-01 01:00:00+01:00.
    """
    df[time_col] = pd.to_datetime(df[time_col], unit='ms')  # Converts from ms to datetime
    if not df[time_col].dt.tz:
        df[time_col] = df[time_col].dt.tz_localize('UTC')  # Localize to UTC if not tz-aware
    df[time_col] = df[time_col].dt.tz_convert('Europe/Berlin')  # Adjustes to Berlin timezone
    return df

def convert_timestamps_from_localized_datetime_to_miliseconds(df: pd.DataFrame, time_col: str) -> pd.DataFrame: 
    """
    Converts the specified time column from timezone-aware datetime in Europe/Berlin 
    to milliseconds since Unix epoch in UTC.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time column as localized datetime
        in the format '%Y-%m-%d %H:%M:%S'.
        time_col (str): Name of the time column to convert.

    Returns:
        pd.DataFrame: Modified DataFrame with the time column in milliseconds since Unix epoch
        e.g., 1609455600000.
    """
    df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S') # Converts to datetime using the specified format

    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize('Europe/Berlin')  # Localize to Berlin if naive
    else:
        df[time_col] = df[time_col].dt.tz_convert('Europe/Berlin')  # Convert to Berlin if already tz-aware

    df[time_col] = df[time_col].dt.tz_convert('UTC')    # Converts the localized datetime to UTC
    
    # Calculates milliseconds since the Unix epoch
    epoch = pd.Timestamp("1970-01-01", tz='UTC')
    df[time_col] = (df[time_col] - epoch) // pd.Timedelta('1ms')
    return df

## srs

def convert_timestamps_from_miliseconds_to_localized_datetime_srs(data_srs: pd.Series) -> pd.Series:
    """
    Converts the specified time column from milliseconds since Unix epoch to 
    timezone-aware datetime in Europe/Berlin timezone.

    Args:
        data_srs (pd.Series): The column in a DataFrame containing the time column in milliseconds.

    Returns:
        pd.Series: Modified time column as localized datetime, e.g., 2000-01-01 01:00:00+01:00.
    """
    data_srs = pd.to_datetime(data_srs, unit='ms')  # Converts from ms to datetime
    if not data_srs.dt.tz:
        data_srs = data_srs.dt.tz_localize('UTC')  # Localize to UTC if not tz-aware
    data_srs = data_srs.dt.tz_convert('Europe/Berlin')  # Adjustes to Berlin timezone
    return data_srs

def convert_timestamps_from_localized_datetime_to_miliseconds_srs(data_srs: pd.Series) -> pd.Series:
    """
    Converts the specified time column from timezone-aware datetime in Europe/Berlin 
    to milliseconds since Unix epoch in UTC.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time column as localized datetime
        in the format '%Y-%m-%d %H:%M:%S'.

    Returns:
        pd.DataFrame: Modified DataFrame with the time column in milliseconds since Unix epoch
        e.g., 1609455600000.
    """
    data_srs = pd.to_datetime(data_srs, format='%Y-%m-%d %H:%M:%S') # Converts to datetime using the specified format

    if data_srs.dt.tz is None:
        data_srs = data_srs.dt.tz_localize('Europe/Berlin')  # Localize to Berlin if naive
    else:
        data_srs = data_srs.dt.tz_convert('Europe/Berlin')  # Convert to Berlin if already tz-aware

    data_srs = data_srs.dt.tz_convert('UTC')    # Converts the localized datetime to UTC
    
    # Calculates milliseconds since the Unix epoch
    epoch = pd.Timestamp("1970-01-01", tz='UTC')
    data_srs = (data_srs - epoch) // pd.Timedelta('1ms')
    return data_srs