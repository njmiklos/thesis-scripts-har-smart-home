import pandas as pd
from typing import List
from pathlib import Path

from get_env import get_input_path, get_output_path, get_annotations_file_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from infer_metadata import infer_data_collection_days_from_time_column
from filter_df import filter_by_timestamp, filter_by_date


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
            save_pandas_dataframe_to_csv(episode.copy(), output_path / filename)
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
        output_file = f'episode_{day}.csv'
        output_file_path = output_path / output_file
        save_pandas_dataframe_to_csv(daily_df, output_file_path)

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
    rows_df_a = int(total_number_rows * (proportion / 100))

    df_a = df.iloc[:rows_df_a]
    df_b = df.iloc[rows_df_a:]

    output_file_a = 'dataset_a.csv'
    output_file_a_path = output_path / output_file_a
    output_file_b = 'dataset_b.csv'
    output_file_b_path = output_path / output_file_b

    save_pandas_dataframe_to_csv(df_a, output_file_a_path)
    save_pandas_dataframe_to_csv(df_b, output_file_b_path)


if __name__ == '__main__':
    # Paths
    annotated_episodes_path = get_annotations_file_path()
    input_file_path = 'synchronized_merged_selected_annotated.csv'
    input_dataset_path = get_input_path() / input_file_path
    output_path = get_output_path()

    # Read datasets
    dataset_df = read_csv_to_pandas_dataframe(input_dataset_path)
    #annotations_df = read_csv_to_pandas_dataframe(annotated_episodes_path)

    # Split into annotated episodes
    #annotated_episodes = segment_into_annotated_episodes(dataset_df, annotations_df)
    # Save episodes
    #save_annotated_episodes(annotated_episodes, output_path)

    segment_df_into_two_parts(dataset_df, 84, output_path)