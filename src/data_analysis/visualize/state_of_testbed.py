"""
This script visualizes synchronized environmental data from CSV logs in a testbed. It processes 
measurements such as temperature, humidity, luminosity, and motion sensor data to generate 
time series plots that illustrate the environmental state during activity episodes.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file to define where CSV input data is read from and where 
  visualizations are saved.
- Input files must be CSVs containing a `time` column and sensor data fields.
- Measurement columns must follow a naming convention that includes measurement type and optionally the location.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
import pandas as pd

from pathlib import Path
from typing import List, Optional

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, get_all_csv_files_in_directory, check_if_output_directory_exists
from data_analysis.visualize.utils import generate_timeseries_plot


def get_column_name(df: pd.DataFrame, measurement: str, location: Optional[str] = None) -> List[str]:
    """
    Retrieves the column name from a DataFrame that matches a given measurement and optional location.

    Args:
        df (pd.DataFrame): The DataFrame to search for matching columns.
        measurement (str): The measurement name to look for (e.g., "temperature").
        location (Optional[str]): The optional location name to narrow down the search.

    Returns:
        List[str]: The name of the matching column.

    Raises:
        ValueError: If no matching or multiple matching columns are found.
    """
    matching_columns = []
    for col in df.columns:
        if measurement in col:
            if not location or location in col:
                matching_columns.append(col)

    if not matching_columns:
        raise ValueError(f'No column found containing measurement "{measurement}"' +
                         (f' and location "{location}".' if location else '.'))

    return matching_columns

def process_single_location_measurements(df: pd.DataFrame, output_dir: Path, measurements: List[str]) -> None:
    """
    Processes and plots time series data for single-location measurements.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        output_dir (Path): Directory where the plots will be saved.
        measurements (List[str]): List of measurement names to plot.

    Returns:
        None
    """
    for measurement in measurements:
        column_names = get_column_name(df, measurement)
        for column_name in column_names:    
            print(f'- Generating plot of {measurement}...')
            generate_timeseries_plot(time_srs=df['time'], data_srs=df[column_name], 
                                        plot_title=column_name, time_axis_label='Date',
                                        value_axis_label=f'{measurement}', output_dir_path=output_dir)

def process_multi_location_measurements(df: pd.DataFrame, output_dir: Path, 
                                        measurements: List[str], locations: List[str]) -> None:
    """
    Processes and plots time series data for measurements available in multiple locations.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        output_dir (Path): Directory where the plots will be saved.
        measurements (List[str]): List of measurement names to plot.
        locations (List[str]): List of location names to combine with measurements.

    Returns:
        None
    """
    for location in locations:
        for measurement in measurements:
            column_names = get_column_name(df, measurement, location)
            for column_name in column_names:
                print(f'- Generating plot of {location} {measurement}...')
                generate_timeseries_plot(time_srs=df['time'], data_srs=df[column_name], 
                                            plot_title=column_name, time_axis_label='Date',
                                            value_axis_label=f'{measurement}', output_dir_path=output_dir)
        
def plot_locations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Processes a DataFrame by generating plots for both single-location and multi-location measurements.

    Args:
        df (pd.DataFrame): The data to be processed and visualized.
        output_dir (Path): Directory where the plots will be saved.

    Returns:
        None
    """
    locations = ['kitchen', 'living room', 'entrance']
    multi_location_measurements = ['humidity', 'luminosity', 'magnitude accelerometer', 
                                   'magnitude gyroscope', 'temperature'] 
    single_location_measurements = ['CO2', 'air quality', 'min sound', 'max sound']

    process_single_location_measurements(df, output_dir, single_location_measurements)
    process_multi_location_measurements(df, output_dir, multi_location_measurements, locations)

def visualize_state_of_testbed(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all CSV files in the input directory by reading, analyzing, and plotting the data.

    Args:
        input_dir (Path): Directory containing input CSV files.
        output_dir (Path): Directory where output plots will be saved.

    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir)
    file_no = len(files)
    counter = 1

    for file in files:
        print(f'Processing {counter}/{file_no} {file.name}...')
        try:
            df = read_csv_to_dataframe(file)

            episode_name = file.stem
            episode_output_dir = output_dir / f'{episode_name}'
            episode_output_dir.mkdir(parents=True, exist_ok=True)

            plot_locations(df, episode_output_dir)
        except Exception as e:
            print(f'Error processing {file}: {e}')
        counter += 1


if __name__ == '__main__':
    input_dir = get_path_from_env('INPUTS_PATH')
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist.')
    
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir)

    visualize_state_of_testbed(input_dir, output_dir)