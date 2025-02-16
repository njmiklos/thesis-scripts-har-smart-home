import pandas as pd

from typing import List
from pathlib import Path

from get_env import get_base_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime, convert_timestamps_from_localized_datetime_to_miliseconds


def check_if_new_episode(row: pd.Series, prev_row: pd.Series, max_time_diff: pd.Timedelta) -> bool:
    """
    Checks if a new episode should start based on the time gap or annotation change.

    Args:
        row (pd.Series): Current row of the DataFrame.
        prev_row (pd.Series): Previous row of the DataFrame.
        max_time_diff (pd.Timedelta): The maximum allowed time difference between consecutive samples to stay in the same episode.

    Returns:
        bool: True if a new episode should start, False if it should not.
    """
    if prev_row is not None:
        time_diff = row['time'] - prev_row['time']
        if (row['annotation'] != prev_row['annotation']) or (time_diff > max_time_diff):
            return True
    
    return False

def find_episodes(df: pd.DataFrame, sampling_rate_in_secs: int) -> List[pd.DataFrame]:
    """
    Identifies and splits the DataFrame into episodes based on time gaps or annotation changes.

    Args:
        df (pd.DataFrame): The DataFrame containing the event data.
        sampling_rate_in_secs (int): The maximum allowed time difference (in seconds) between consecutive samples to stay in the same episode.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing a separate episode.
    """
    max_time_diff = pd.Timedelta(seconds=sampling_rate_in_secs)

    episodes = []
    current_episode = []
    prev_row = None

    for _, row in df.iterrows():
        if check_if_new_episode(row, prev_row, max_time_diff):
            if current_episode:
                episodes.append(pd.DataFrame(current_episode))
            current_episode = []

        current_episode.append(row)
        prev_row = row

    if current_episode:
        episodes.append(pd.DataFrame(current_episode))
    
    return episodes

def save_episodes(episodes: List[pd.DataFrame], output_path: Path) -> None:
    """
    Saves each episode in the list to a separate CSV file.

    Args:
        episodes (List[pd.DataFrame]): A list of DataFrames, each representing an episode.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    for i, episode in enumerate(episodes):
        annotation_name = episode['annotation'].iloc[0].replace(' ', '_')
        filename = f"ep_{i + 1}_{annotation_name}.csv"

        episode = convert_timestamps_from_localized_datetime_to_miliseconds(episode, 'time')

        save_pandas_dataframe_to_csv(episode, output_path / filename)
        print(f"Saved {filename}")


if __name__ == '__main__':
    base_path = get_base_path()
    dataset_path = base_path / 'Dataset Transition Activities'
    dataset_file = dataset_path / 'Transition Activities.csv'
    sampling_rate_in_secs = 10

    df = read_csv_to_pandas_dataframe(dataset_file)
    df = convert_timestamps_from_miliseconds_to_localized_datetime(df, 'time')

    episodes = find_episodes(df, sampling_rate_in_secs)

    save_episodes(episodes, dataset_path)
