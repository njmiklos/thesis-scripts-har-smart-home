"""
Compresses raw sensor data into a textual summary that higlights changes in sensor data.
"""

import pandas as pd

from pathlib import Path
from typing import List, Tuple

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, get_all_csv_files_in_directory
from data_processing.convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime_srs


def remove_consecutive_duplicates(series: pd.Series) -> List[float]:
    """
    Removes consecutive duplicate values from a pandas Series.

    Args:
        series (pd.Series): The input data column.

    Returns:
        List[float]: List of values with consecutive duplicates removed.
    """
    values = [series.iloc[0]]
    for value in series.iloc[1:]:
        if value != values[-1]:
            values.append(value)
    return values

def compress_column(series: pd.Series) -> List[float]:
    """
    Compresses a column by removing consecutive duplicates and rounding values to two decimal positions.

    Args:
        series (pd.Series): The input column.

    Returns:
        List[float]: A compressed version of the measurements.
    """
    rounded_series = series.round(2)
    compressed = remove_consecutive_duplicates(rounded_series)
    return compressed

def parse_column_name(name: str) -> Tuple[str, str]:
    """
    Extracts location and measurement name from a column title.

    Args:
        name (str): Column name in the format 'location measurement [unit]'.

    Returns:
        Tuple[str, str]: A tuple containing the location (e.g., 'Kitchen') and the measurement string.
    """
    if name == 'time':
        return 'time', 'time'

    known_locations = ['kitchen', 'entrance', 'living room']

    for loc in known_locations:
        if name.startswith(loc):
            measurement = name[len(loc):].strip()
            return loc.title(), measurement

    return 'Unknown', name

def get_time(df: pd.DataFrame) -> str:
    """
    Extracts the time of day from the first timestamp in the DataFrame. 
    
    Args:
        df (pd.DataFrame): Input DataFrame with a 'time' column in miliseconds UTC Europe/Berlin.

    Returns:
        str: Time in HH:MM format.
    """
    df = df.copy()
    df['time'] = convert_timestamps_from_miliseconds_to_localized_datetime_srs(df['time'])
    first_timestamp = df['time'].iloc[0]
    time_str = first_timestamp.strftime('%H:%M')
    return time_str

def group_sensor_data(df: pd.DataFrame) -> dict[str, List[str]]:
    """
    Groups and compresses sensor data by location and measurement.

    Args:
        df (pd.DataFrame): The input sensor DataFrame.

    Returns:
        dict[str, List[str]]: Dictionary mapping location names to sensor summaries.
    """
    grouped_data: dict[str, List[str]] = {}

    for column in df.columns:
        if column == 'time':
            continue
        location, measurement = parse_column_name(column)
        compressed_values = compress_column(df[column])
        grouped_data.setdefault(location, []).append(f'  {measurement}: {compressed_values}')

    return grouped_data

def format_summary(time_str: str, grouped_data: dict[str, List[str]]) -> str:
    """
    Formats the compressed sensor data into a human-readable text summary.

    Args:
        time_str (str): Time of day in HH:MM format.
        grouped_data (dict[str, List[str]]): Grouped sensor data by location.

    Returns:
        str: A formatted summary.
    """
    output: List[str] = [f'Time: {time_str}\n']

    for location in sorted(grouped_data.keys()):
        output.append(f'{location}')
        output.extend(grouped_data[location])
        output.append('')

    return '\n'.join(output)

def generate_summary(df: pd.DataFrame) -> str:
    """
    Generates a compact, human-readable summary of sensor data from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series sensor data.

    Returns:
        str: Summary of compressed sensor values.
    """
    if 'annotation' in df.columns:
        df.drop(columns=['annotation'], inplace=True)

    time_str = get_time(df)
    grouped_data = group_sensor_data(df)
    return format_summary(time_str, grouped_data)

def write_to_text_file(output_path: Path, text: str):
    """
    Writes a string to a text file.

    Args:
        output_path (Path): Path to the output file.
        text (str): Text content to be written.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_files(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all CSV files in the input directory by generating summaries
    and writing them to corresponding text files in the output directory.

    Args:
        input_dir (Path): Directory containing input CSV files.
        output_dir (Path): Directory to write output summary files.
    """
    files = get_all_csv_files_in_directory(input_dir)
    for file in files:
        df = read_csv_to_dataframe(file)
        summary_text = generate_summary(df)

        filename = file.stem + '.txt'
        output_path = output_dir / filename
        write_to_text_file(output_path, summary_text)


if __name__ == '__main__':
    input_path = get_path_from_env('INPUTS_PATH')
    output_path = get_path_from_env('OUTPUTS_PATH')
    output_path.mkdir(parents=True, exist_ok=True)

    process_files(input_path, output_path)