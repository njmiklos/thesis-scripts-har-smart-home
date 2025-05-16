"""
Provides utilities for segmenting time-series sensor data into meaningful chunks,
such as annotated episodes, daily files, fixed-length windows, or arbitrary slices around specific 
timestamps or row ranges.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in a `.env` file.
- Input files must include a `time` column in milliseconds.
- Refer to `README.md` for full setup and usage instructions.
"""
import pandas as pd
from typing import List
from pathlib import Path

from utils.get_env import get_path_from_env
from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv, check_if_directory_exists
from data_processing.infer.metadata import infer_data_collection_days_from_time_column
from data_processing.filter import filter_by_timestamp, filter_by_date


def segment_into_annotated_episodes(dataset_df: pd.DataFrame, annotations_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Splits sensor data into episodes based on the list of the annotated episodes.

    Args:
        dataset_df (pd.DataFrame): The DataFrame containing the event data.
        annotations_df (pd.DataFrame): The DataFrame containing annotated episodes.

    Returns:
        List[pd.DataFrame]: A list of episode DataFrames.
    """
    unannotated = False
    if 'annotation' not in dataset_df.columns:
        dataset_df['annotation'] = None
        unannotated = True

    episodes = []

    for _, row in annotations_df.iterrows():
        start_time = row['start']
        end_time = row['end']
        annotation = row['annotation']

        episode_df = filter_by_timestamp(dataset_df, start_time, end_time).copy()

        if not episode_df.empty:
            if unannotated:
                episode_df.loc[:, 'annotation'] = annotation
            episodes.append(episode_df)

    return episodes

def get_filename(episode_number: int, episode: pd.DataFrame) -> str:
    """
    Generates a descriptive filename for an annotated episode CSV file.

    Args:
        episode_number (int): The sequential number of the episode.
        episode (pd.DataFrame): A DataFrame containing episode data, including 'time' (in ms UTC) and 'annotation'.

    Returns:
        str: The generated filename in the format 'YYYY-MM-DD_ep_<episode_number>_<annotation>.csv'.
    """
    if episode.empty:
        raise ValueError(f'Episode {episode_number} is empty and has no timestamp or annotation.')

    cls = str(episode['annotation'].iloc[0])
    cls = cls.replace(' ', '_')

    day = pd.to_datetime(episode['time'].iloc[0], unit='ms', utc=True).tz_convert('Europe/Berlin')
    day_str = day.strftime('%Y-%m-%d')

    filename = f'{day_str}_ep_{episode_number}_{cls}.csv'
    return filename

def save_annotated_episodes(episodes: List[pd.DataFrame], output_path: Path) -> None:
    """
    Saves each annotated episode in the list to a separate CSV file.

    Args:
        episodes (List[pd.DataFrame]): A list of episode DataFrames.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for i, episode in enumerate(episodes):
        try:
            filename = get_filename(i + 1, episode)
            save_dataframe_to_csv(episode.copy(), output_path / filename)
            print(f'Saved {filename}')
        except Exception as e:
            print(f'Failed to save episode {i + 1}: {e}')

def segment_df_into_days(df: pd.DataFrame, output_path: Path):
    """
    Saves each day in the dataset to a separate CSV file.

    Args:
        dataset_df (pd.DataFrame): The DataFrame containing all activity data.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    days = infer_data_collection_days_from_time_column(df['time'])

    for day in days:
        daily_df = filter_by_date(df, day)
        output_file = f'day_{day}.csv'
        output_file_path = output_path / output_file
        save_dataframe_to_csv(daily_df, output_file_path)

def segment_df_into_two_parts(df: pd.DataFrame, proportion: int, output_path: Path):
    """
    Splits the DataFrame into two segments based on the specified proportion, and saves them as separate CSV files.

    The first CSV file (dataset_a.csv) will contain the first 'proportion' percent of the rows, 
    and the second CSV file (dataset_b.csv) will contain the remaining rows.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be segmented.
        proportion (int): The percentage (between 0 and 100) of rows to include in the first CSV file.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    if not (0 < proportion < 100):
        raise ValueError('Proportion must be between 0 and 100 (exclusive).')

    total_number_rows = len(df)
    if total_number_rows == 0:
        raise ValueError('The DataFrame is empty.')
    
    rows_df_a = int(total_number_rows * (proportion / 100))

    df_a = df.iloc[:rows_df_a]
    df_b = df.iloc[rows_df_a:]

    output_file_a = 'dataset_a.csv'
    output_file_a_path = output_path / output_file_a
    output_file_b = 'dataset_b.csv'
    output_file_b_path = output_path / output_file_b

    save_dataframe_to_csv(df_a, output_file_a_path)
    save_dataframe_to_csv(df_b, output_file_b_path)

def segment_from_row_position_to_row_position(df: pd.DataFrame, start_position: int, end_position: int, output_path: Path, output_filename: str = 'slice.csv') -> None:
    """
    Saves into a file a segment of the data between the specified row positions in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be segmented.
        start_position (int):  The starting row position (inclusive).
        end_position (int):  The ending row position (exclusive).
        output_path (Path): The directory where the CSV files will be saved.
        output_filename (str): The name of the output CSV file. Defaults to 'slice.csv'.

    Returns:
        None
    """
    new_df = df.iloc[start_position : end_position]
    save_dataframe_to_csv(new_df, output_path / output_filename)

def extract_window_around_timestamp(df: pd.DataFrame, window_size: int, timestamp: int, output_path: Path, output_filename: str = 'slice.csv') -> None:
    """
    Extracts and saves a segment of the DataFrame with the specified window size,
    centered around the specified timestamp.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The number of rows to include in the extracted segment.
        timestamp (int): The timestamp in the centre of the window to be.
        output_path (Path): The directory where the CSV file will be saved.
        output_filename (str, optional): The name of the output CSV file. Defaults to 'slice.csv'.
    """
    total_number_rows = len(df)
    if total_number_rows == 0:
        raise ValueError('The DataFrame is empty.')

    if window_size >= total_number_rows:
        raise ValueError('The window size needs to be smaller than the DataFrame.')

    if window_size < 1:
        raise ValueError('The window size should be at least 1 sample.')

    #middle_positions = df.index[df['time'] == timestamp].tolist()
    matches = df['time'] == timestamp   # boolean series with True where time column equals timestamp
    matching_indices = df.index[matches]    # index positions where time column equals timestamp
    middle_positions = matching_indices.tolist()    # list of middle positions

    if not middle_positions:
        raise ValueError(f'Timestamp {timestamp} not found in the DataFrame.')

    middle_position = middle_positions[0]  # Take first occurrence if multiple

    start_position = middle_position - window_size // 2
    if start_position < 0:
        raise ValueError('Cannot segment with this window, window start is out of bounds.')

    end_position = start_position + window_size
    if end_position > total_number_rows:
        raise ValueError('Cannot segment with this window, end of window is out of bounds.')

    segment_from_row_position_to_row_position(df, start_position, end_position, output_path, output_filename)

def extract_middle_segment(df: pd.DataFrame, window_size: int, output_path: Path, output_filename: str = 'slice.csv') -> None:
    """
    Extracts and saves a middle segment of the DataFrame with the specified window size.
    
    The extracted segment is centered around the middle sample of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The number of rows to include in the extracted segment.
        output_path (Path): The directory where the CSV file will be saved.
        output_filename (str, optional): The name of the output CSV file. Defaults to 'slice.csv'.
    """
    total_number_rows = len(df)
    if total_number_rows == 0:
        raise ValueError('The DataFrame is empty.')
    
    if total_number_rows < 3:
        raise ValueError('The DataFrame has less than 3 rows, there is no middle sample.')
    
    if window_size >= total_number_rows:
        raise ValueError('The window size needs to be smaller than the DataFrame.')

    if window_size < 1:
        raise ValueError('The window size should be at least 1 sample.')

    middle_position = total_number_rows // 2

    start_position = middle_position - window_size // 2
    if start_position < 0:
        raise ValueError('Cannot segment with this window, window start is out of bounds.')
    
    end_position = start_position + window_size
    if end_position > total_number_rows:
        raise ValueError('Cannot segment with this window, end of window is out of bounds.')

    segment_from_row_position_to_row_position(df, start_position, end_position, output_path, output_filename)

def segment_into_windows(df: pd.DataFrame, window_size: int, overlap_size: int, output_directory_path: Path, output_filename_prefix: str = 'window'):
    """
    Segments a DataFrame into overlapping windows and saves each window as a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be segmented.
        window_size (int): The number of rows in each segmented window.
        overlap_size (int): The number of overlapping rows between consecutive windows.
        output_directory (Path): The directory where the segmented CSV files will be saved.
        output_filename_prefix (str, optional): The prefix for the output CSV filenames. Defaults to 'window'.

    Returns:
        None
    """
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError('The input DataFrame is empty. There is no data to segment.')
    
    if window_size > total_rows:
        raise ValueError(f'Invalid window size: {window_size}. The DataFrame contains only {total_rows} rows, so no segmentation is possible.')
    
    if window_size < 1:
        raise ValueError(f'Invalid window size: {window_size}. A segment must contain at least one row.')
    
    if overlap_size < 0:
        raise ValueError(f'Invalid overlap size: {overlap_size}. Overlap must be 0 or greater.')

    if overlap_size >= window_size:
        raise ValueError(f'Invalid overlap size: {overlap_size}. The overlap must be smaller than the window size ({window_size}).')

    start_position = 0
    segment_count = 1
    while start_position + window_size <= total_rows:
        end_position = start_position + window_size
        output_filename = f'{output_filename_prefix}_{segment_count}.csv'
        segment_from_row_position_to_row_position(df, start_position, end_position, output_directory_path, output_filename)
        
        start_position += (window_size - overlap_size)
        segment_count += 1


if __name__ == '__main__':
    # Paths
    #annotated_episodes_path = get_path_from_env('ANNOTATIONS_FILE_PATH')
    input_file_name = 'synchronized_merged_selected_annotated_new_col_names.csv'

    input_dataset_path = get_path_from_env('INPUTS_PATH') / input_file_name
    output_path = get_path_from_env('OUTPUTS_PATH')
    check_if_directory_exists(output_path)

    # Read datasets
    dataset_df = read_csv_to_dataframe(input_dataset_path)
    #annotations_df = read_csv_to_pandas_dataframe(annotated_episodes_path)

    # Split into annotated episodes
    #annotated_episodes = segment_into_annotated_episodes(dataset_df, annotations_df)
    # Save episodes
    #save_annotated_episodes(annotated_episodes, output_path)

    # Save a segment
    #extract_middle_segment(dataset_df, 300, output_path, input_file_name)
    #segment_from_row_position_to_row_position(dataset_df, 413, 714, output_path, input_file_name)

    episodes = ['Airing', 'Eating a meal', 'Eating breakfast', 'Eating dinner', 'Eating supper',
                'Entering home', 'Getting up', 'Leaving home', 'Other', 'Preparing a drink', 
                'Preparing a meal', 'Preparing breakfast', 'Preparing dinner', 'Preparing for bed', 'Preparing supper', 
                'Relaxing', 'Sleeping', 'Working', 'Working out']
    timestamps = [1733952556000, 1733415268000, 1733724929000, 1733568599000, 1734461489000, 
                  1734168749000, 1733377106000, 1733816039000, 1733929049000, 1733833948000, 
                  1733413809000, 1733723175000, 1733566648000, 1734039915000, 1734458602000, 
                  1734294629000, 1733794789000, 1733388388000, 1733933998000]
    
    for episode, timestamp in zip(episodes, timestamps):
        output_filename  =  f'{episode}.csv'
        extract_window_around_timestamp(dataset_df, 900, timestamp, output_path, output_filename)