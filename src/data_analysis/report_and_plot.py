import pandas as pd
from pathlib import Path

from typing import List, Tuple, Optional

from utils.get_env import get_path_from_env
from data_processing.infer.sensor_metadata import infer_unit
from utils.handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv, get_all_csv_files_in_directory
from data_analysis.report_utils import get_quick_stats_dict, get_root_mean_square_error_srs
from data_analysis.visualize.utils import generate_histogram, generate_scatter_plot, generate_comparative_scatterplots


def get_labels(device_type: str, measurement: str, modification: str) -> Tuple:
    """
    Generates labels for the given device, measurement, and modification.

    Args:
        device_type (str): The type of the device or sensor.
        measurement (str): The measurement name (e.g., 'humidity', 'temperature').
        modification (str): The modification applied (e.g., 'Raw', 'resampled').

    Returns:
        Tuple: A tuple containing the following elements:
            - title (str): The title for the plot.
            - time_axis_label (str): The label for the time axis.
            - value_axis_label (str): The label for the value axis.
    """
    title = f'{device_type} {measurement} - {modification}'
    time_axis_label = 'Date'
    unit = infer_unit(measurement)
    value_axis_label = f'{measurement} [{unit}]'
    return title, time_axis_label, value_axis_label

def get_df_sample(file_path: Path, sample_size: float) -> pd.DataFrame:
    """
    Reads a CSV file into a Pandas DataFrame and returns a sampled fraction of it.

    Args:
        file_path (Path): Path to the CSV file.
        sample_size (float): Fraction of the dataset to sample (e.g., 0.1 for 10%).

    Returns:
        pd.DataFrame: A sampled DataFrame if the file is not empty; otherwise, an empty DataFrame.
    """
    df = read_csv_to_pandas_dataframe(file_path)
    if not df.empty:
        df = df.sample(frac=sample_size, random_state=42)
    else:
        print('WARNING: Got an empty DataFrame, cannot sample that, skipping.')
    return df

def get_sampling_sizes(sampling: bool, file_path: Path, motion_size: float, ambient_size: float) -> float:
    """
    Determines the appropriate sampling size based on file type and user-defined settings.

    Args:
        sampling (bool): Whether to apply sampling.
        file_path (Path): Path to the dataset file.
        motion_size (float): Sampling fraction for motion-related data.
        ambient_size (float): Sampling fraction for ambient-related data.

    Returns:
        float: The appropriate sample size for the given dataset.
    """
    if sampling:
        if 'magnitude' in str(file_path.stem) or 'motion' in str(file_path.stem):
            sample_size = motion_size
        else:
            sample_size = ambient_size
    else:
        sample_size = 1.0   # 100%
    return sample_size

def generate_plots(output_path: Path, histogram: bool, scatter: bool,
                   time_srs1: pd.Series, data_srs1: pd.Series,
                   time_srs2: Optional[pd.Series], data_srs2: Optional[pd.Series], modification: Optional[str],
                   title: str, time_axis_label: str, value_axis_label: str) -> None:
    """
    Generates and saves plots based on the provided data series.

    Depending on the flags provided, this function will:
      - Generate a histogram using the data series (preferring data_srs2 if available, otherwise data_srs1).
      - Generate a scatter plot. If both a secondary time series (time_srs2) and data series (data_srs2)
        are provided along with a modification label, it creates a comparative scatter plot; otherwise,
        it generates a standard scatter plot using time_srs1 and data_srs1.

    Args:
        output_path (Path): The directory where the plot files will be saved.
        histogram (bool): Whether to generate a histogram.
        scatter (bool): Whether to generate a scatter plot.
        time_srs1 (pd.Series): The primary time series data.
        data_srs1 (pd.Series): The primary data series.
        time_srs2 (Optional[pd.Series]): The secondary time series data (for comparisons). Defaults to None.
        data_srs2 (Optional[pd.Series]): The secondary data series (for comparisons). Defaults to None.
        modification (Optional[str]): A label for the modification type, used in the comparative plot.
        title (str): The title for the plot.
        time_axis_label (str): The label for the time (x) axis.
        value_axis_label (str): The label for the value (y) axis.

    Returns:
        None
    """
    if histogram:
        if not time_srs2.empty and not data_srs2.empty:
            success = generate_histogram(data_srs2, title, 'Frequency', value_axis_label, output_path)
        else:
            success = generate_histogram(data_srs1, title, 'Frequency', value_axis_label, output_path)

        if success:
            print(f'Saved a histogram "{title}" to {output_path}.')
        else:
            print(f'Could not generate a histogram "{title}".')

    if scatter:
        if not time_srs2.empty and not data_srs2.empty and modification:
            success = generate_comparative_scatterplots(time_srs1, time_srs2, data_srs1, data_srs2, modification, 
                                                title, time_axis_label, value_axis_label, output_path)
        else:
            success = generate_scatter_plot(time_srs1, data_srs1, title, time_axis_label, value_axis_label, output_path)

        if success:
            print(f'Saved a scatter plot "{title}" to {output_path}.')
        else:
            print(f'Could not generate a scatter plot "{title}".')

def save_summary(summary_df: pd.DataFrame, output_path: Path, modification: str = ''):
    """
    Saves the summary DataFrame to a CSV file in the specified output directory.

    If the summary DataFrame is not empty, the function saves it using a filename that reflects the 
    provided modification type. When a modification label is provided, the file is named 
    "stats_summary_{modification}.csv"; otherwise, it defaults to "stats_summary_raw.csv". The function 
    prints a message indicating the path and name of the saved file. If the DataFrame is empty, a message 
    is printed indicating that there is no data to save.

    Args:
        summary_df (pd.DataFrame): The DataFrame containing summary statistics.
        output_path (Path): The directory where the summary CSV file will be saved.
        modification (str, optional): A label indicating the type of modification. Defaults to an empty 
                                      string, resulting in the file being named "stats_summary_raw.csv".

    Returns:
        None
    """
    if not summary_df.empty:
        if modification:
            modification = modification.lower()
            print(f'Saving modified summary file to {output_path}/stats_summary_{modification}.csv')
            save_pandas_dataframe_to_csv(summary_df, output_path / f'stats_summary_{modification}.csv')
        else:
            print(f'Saving modified summary file to {output_path}/stats_summary_raw.csv')
            save_pandas_dataframe_to_csv(summary_df, output_path / f'stats_summary_raw.csv')
    else:
        print('No data to save.')

def get_dataframes(sampling: bool, og_file_path: Path, mod_file_path: Optional[Path], sample_size_motion: Optional[float] = 0.1, 
                   sample_size_ambient: Optional[float] = 0.5) -> Tuple[pd. DataFrame, Optional[pd.DataFrame]]:
    """
    Reads CSV files for the original dataset and, optionally, a modified dataset, applying sampling if enabled.

    This function returns a tuple where:
      - The first element is a DataFrame for the original file.
      - The second element is a DataFrame for the modified file if a modified file path is provided; otherwise, it returns None.

    When sampling is enabled, the appropriate sampling fraction is determined based on whether the file is
    related to motion data (using sample_size_motion) or ambient data (using sample_size_ambient).

    Args:
        sampling (bool): Whether to apply sampling to the data.
        og_file_path (Path): Path to the original CSV file.
        mod_file_path (Optional[Path]): Path to the modified CSV file, if available.
        sample_size_motion (Optional[float]): Sampling fraction for motion-related data (required if sampling is True).
        sample_size_ambient (Optional[float]): Sampling fraction for ambient-related data (required if sampling is True).

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            - The first element is the DataFrame read from the original file.
            - The second element is the DataFrame read from the modified file if provided; otherwise, None.
    """
    mod_file_df = None

    if sampling and sample_size_motion and sample_size_ambient:
        sample_size = get_sampling_sizes(sampling, og_file_path, sample_size_motion, sample_size_ambient)
        print(f'Using {sample_size * 100}% of data for analysis.')

        og_file_df = get_df_sample(og_file_path, sample_size)
        if mod_file_path:
            mod_file_df = get_df_sample(mod_file_path, sample_size)
        
    else:
        og_file_df = read_csv_to_pandas_dataframe(og_file_path)
        if mod_file_path:
            mod_file_df = read_csv_to_pandas_dataframe(mod_file_path)

    return og_file_df, mod_file_df

def get_new_table_for_summary(file_paths: List[Path], output_path: Path, histogram: bool, scatter: bool, 
                                sampling: bool, sample_size_motion: float, sample_size_ambient: float) -> None:
    """
    Creates and saves a new summary CSV file containing statistical metrics for each measurement column 
    (excluding the 'time' column) from the provided CSV files.

    For each file in the list, the function:
      - Reads the CSV data (applying sampling if enabled).
      - Iterates over each measurement column (skipping the 'time' column).
      - Computes quick statistics using `get_quick_stats_dict` for each measurement.
      - Optionally generates histogram and scatter plots based on the provided flags.
      - Populates a summary DataFrame with the computed statistics under the "Raw" column.
      - Inserts a placeholder (None) for the RMSE row for each measurement.

    Finally, the summary DataFrame is saved as a CSV file in the specified output directory with the filename 
    "stats_summary_raw.csv".

    Args:
        file_paths (List[Path]): List of file paths to the CSV files containing raw data.
        output_path (Path): Directory where the summary CSV file will be saved.
        histogram (bool): Flag indicating whether to generate histogram plots for each measurement.
        scatter (bool): Flag indicating whether to generate scatter plots for each measurement.
        sampling (bool): Flag indicating whether to apply sampling to the data.
        sample_size_motion (float): Sampling fraction for motion-related data.
        sample_size_ambient (float): Sampling fraction for ambient-related data.

    Returns:
        None
    """
    file_paths_counter = len(file_paths)
    counter = 1
    summary_df = pd.DataFrame()

    for file_path in file_paths:
        print(f'Processing file {counter}/{file_paths_counter} {file_path.stem}...')
        device_type = file_path.stem
        device_type = device_type.replace('_', ' ')

        file_df, _ = get_dataframes(sampling, file_path, None, sample_size_motion, sample_size_ambient)

        if not file_df.empty:
            for measurement in file_df:
                if measurement != 'time':
                    title, time_axis_label, value_axis_label = get_labels(device_type, measurement, 'Raw')

                    if histogram or scatter:
                        generate_plots(output_path=output_path / 'Graphs', histogram=histogram, scatter=scatter, 
                                    time_srs1=file_df['time'], data_srs1=file_df[measurement],
                                    time_srs2=None, data_srs2=None, modification=None,
                                    title=title, time_axis_label=time_axis_label, value_axis_label=value_axis_label)
                    
                    stats = get_quick_stats_dict(file_df[measurement])
                    for position in stats:
                        position_readable = position.replace('_', ' ')
                        row = f'{device_type} {measurement} {position_readable}'
                        summary_df.at[row, 'Raw'] = stats[position]

        print(f'Processed file {file_path.stem} and collected stats.')
        counter += 1

    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={'index': 'Measurement'})
    save_summary(summary_df, output_path / 'Analysis', modification)

def add_column_to_table_summary(og_dataset_file_paths: List[Path], mod_dataset_file_paths: List[Path], 
                                mod_dataset_path: Path, summary_df: pd.DataFrame, modification: str,
                                histogram: bool, scatter: bool, 
                                sampling: bool, sample_size_motion: float, sample_size_ambient: float) -> None:
    """
    Updates an existing summary CSV file by adding a new column with statistical metrics computed from the 
    modified dataset CSV files.

    For each corresponding pair of original and modified files, the function:
      - Reads both CSV files (applying sampling if enabled).
      - Iterates over each measurement column (skipping the 'time' column).
      - Computes quick statistics using `get_quick_stats_dict` for the modified data and inserts the results 
        into the summary DataFrame under a new column labeled with the modification type.
      - Optionally generates comparative histogram and scatter plots if the respective flags are enabled.
      - Computes the RMSE (root mean square error) between the original and modified measurements and adds this 
        metric to the summary DataFrame.

    The updated summary DataFrame is then saved as a CSV file in the modified dataset directory with a filename 
    that reflects the modification type (e.g., "stats_summary_{modification}.csv").

    Args:
        og_dataset_file_paths (List[Path]): List of file paths for the original dataset CSV files.
        mod_dataset_file_paths (List[Path]): List of file paths for the modified dataset CSV files.
        mod_dataset_path (Path): Directory where the updated summary CSV file will be saved.
        summary_df (pd.DataFrame): The existing summary DataFrame to which the new column will be added.
        modification (str): A label indicating the type of modification applied, which will be used as the column name.
        histogram (bool): Flag indicating whether to generate histogram plots comparing the original and modified data.
        scatter (bool): Flag indicating whether to generate scatter plots comparing the original and modified data.
        sampling (bool): Flag indicating whether to apply sampling to the data.
        sample_size_motion (float): Sampling fraction for motion-related data.
        sample_size_ambient (float): Sampling fraction for ambient-related data.

    Returns:
        None
    """
    output_path = mod_dataset_path
    file_paths_counter = len(mod_dataset_file_paths)
    counter = 1

    for og_file_path, mod_file_path in zip(og_dataset_file_paths, mod_dataset_file_paths):
        print(f'Processing file {counter}/{file_paths_counter} {mod_file_path.stem}...')
        device_type = mod_file_path.stem
        device_type = device_type.replace('_', ' ')

        og_file_df, mod_file_df = get_dataframes(sampling, og_file_path, mod_file_path, 
                                                 sample_size_motion, sample_size_ambient)

        if not summary_df.empty and not mod_file_df.empty:
            for measurement in mod_file_df:
                if measurement != 'time':
                    title, time_axis_label, value_axis_label = get_labels(device_type, measurement, modification)

                    if histogram or scatter:
                        generate_plots(output_path=output_path / 'Graphs', histogram=histogram, scatter=scatter, 
                                    time_srs1=og_file_df['time'], data_srs1=og_file_df[measurement],
                                    time_srs2=mod_file_df['time'], data_srs2=mod_file_df[measurement], modification=modification,
                                    title=title, time_axis_label=time_axis_label, value_axis_label=value_axis_label)
                    
                    stats = get_quick_stats_dict(mod_file_df[measurement])
                    for position in stats:
                        position_readable = position.replace('_', ' ')
                        row = f'{device_type} {measurement} {position_readable}'
                        summary_df.loc[row, modification] = stats[position]

                    rsme = get_root_mean_square_error_srs(og_file_df[measurement], mod_file_df[measurement])
                    rmse_row = f'{device_type} {measurement} RMSE (vs. Raw)'
                    summary_df.loc[rmse_row, modification] = rsme

        print(f'Processed file {mod_file_path.stem} and added modification.')
        counter += 1

    summary_df = summary_df.reset_index()
    summary_df = summary_df.rename(columns={'index': 'Measurement'})
    save_summary(summary_df, output_path / 'Analysis', modification)

def check_if_directory_exists(directory: Path) -> None:
    """
    Makes sure the specified directory exists.
    """
    if not directory.exists() or not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
        print(f'Required directory was missing, created {directory}.')


if __name__ == "__main__":
    base_path = get_path_from_env('BASE_PATH')

    # Set before running
    modification = 'Synchronized'
    og_dataset_path = base_path / 'Raw_relevant'
    #mod_dataset_path = og_dataset_path
    mod_dataset_path = base_path / 'Synchronized'
    histogram = False
    scatter = True
    sampling = False
    sample_size_motion = 0.1
    sample_size_ambient = 0.5

    og_dataset_file_paths = get_all_csv_files_in_directory(og_dataset_path)
    og_dataset_file_paths.sort()

    analysis_filename = 'stats_summary.csv'
    analysis_dir = og_dataset_path / 'Analysis'
    analysis_path = analysis_dir / analysis_filename
    check_if_directory_exists(analysis_dir)

    if mod_dataset_path == og_dataset_path:
        if histogram or scatter:
            graphs_dir = og_dataset_path / 'Graphs'
            check_if_directory_exists(graphs_dir)

        get_new_table_for_summary(og_dataset_file_paths, og_dataset_path, histogram, scatter,
                                  sampling, sample_size_motion, sample_size_ambient)
        
    else:
        analysis_filename = f'table_summary_{modification}.csv'

        mod_analysis_dir = mod_dataset_path / 'Analysis'
        check_if_directory_exists(mod_analysis_dir)

        graphs_dir = mod_dataset_path / 'Graphs'
        check_if_directory_exists(graphs_dir)

        mod_dataset_file_paths = get_all_csv_files_in_directory(mod_dataset_path)
        mod_dataset_file_paths.sort()

        summary_df = read_csv_to_pandas_dataframe(analysis_path)
        add_column_to_table_summary(og_dataset_file_paths, mod_dataset_file_paths, mod_dataset_path, 
                                    summary_df, modification, histogram, scatter,
                                    sampling, sample_size_motion, sample_size_ambient)