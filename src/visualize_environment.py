"""
Plots synchronized data from a household to show the state of the environment,
e.g., during a single activity episode. It compacts the same measurement type 
from different locations to a single graph.
"""
import pandas as pd

from pathlib import Path
from typing import List

from get_env import (get_input_path, get_output_path)
from handle_csv import (read_csv_to_pandas_dataframe, get_all_csv_files_in_directory)
from infer_sensor_metadata import infer_unit
from visualize_data import generate_timeseries_plot, generate_comparative_timeseries_plot_of_3_measurements

def get_column_name_with_measurement(df: pd.DataFrame, measurement: str) -> str:
    column_name_with_measurement = None
    for col in df.columns:
        if measurement in col:
            column_name_with_measurement = col

    if column_name_with_measurement:
        return column_name_with_measurement
    else:
        raise ValueError(f'No column with measurement {measurement} in the DataFrame.')

def process_single_location_measurements(df: pd.DataFrame, output_dir: Path, measurements: List) -> None:
    for measurement in measurements:
        column_name_with_measurement = get_column_name_with_measurement(df, measurement)
        unit = infer_unit(measurement)
        generate_timeseries_plot(time_srs=df['time'], data_srs=df[column_name_with_measurement], 
                                    plot_title=column_name_with_measurement, time_axis_label='Date',
                                    value_axis_label=f'{measurement} [{unit}]', output_dir_path=output_dir)

def get_column_name_with_measurement_and_location(df: pd.DataFrame, measurement: str, location: str) -> str:
    column_name_with_measurement = None
    for col in df.columns:
        if measurement in col:
            column_name_with_measurement = col

    if column_name_with_measurement:
        # find location
    else:
        raise ValueError(f'No column with measurement {measurement} in the DataFrame.')
    
def process_multi_location_measurements(df: pd.DataFrame, output_dir: Path, 
                                        measurements: List[str], locations: List[str]) -> None:
    for location in locations:
        for measurement in measurements:
            column_name_with_measurement_and_location = get_column_name_with_measurement_and_location(df, measurement)
            unit = infer_unit(measurement)

def process_df(df: pd.DataFrame, output_dir: Path) -> None:
    locations = ['kitchen', 'living room', 'entrance']
    multi_location_measurements = ['humidity', 'luminosity', 'magnitude accelerometer', 
                                   'magnitude gyroscope', 'temperature'] 
    single_location_measurements = ['CO2', 'air quality', 'min sound', 'max sound']

    process_single_location_measurements(df, output_dir, single_location_measurements)
    process_multi_location_measurements(df, output_dir, multi_location_measurements, locations)

def process_files(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all CSV files in the input directory.

    Args:
        input_dir (Path): Directory containing input CSV files.
    
    Returns:
        None
    """
    files = get_all_csv_files_in_directory(input_dir)
    for file in files:
        df = read_csv_to_pandas_dataframe(file)
        process_df(df, output_dir)


if __name__ == '__main__':
    input_path = get_input_path()
    output_path = get_output_path()

    process_files(input_path, output_path)