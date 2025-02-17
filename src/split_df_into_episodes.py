import pandas as pd
from typing import List
from pathlib import Path

from get_env import get_input_path, get_output_path, get_annotations_file_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from filter_df import filter_by_timestamp


def split_into_annotated_episodes(dataset_df: pd.DataFrame, annotations_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Splits sensor data into episodes based on the list of the annotated episodes.

    Args:
        dataset_df (pd.DataFrame): The DataFrame containing the event data.
        annotations_df (pd.DataFrame): The DataFrame containing annotated episodes.

    Returns:
        List[pd.DataFrame]: A list of episode DataFrames.
    """
    episodes = []

    for _, row in annotations_df.iterrows():
        start_time = row['start']
        end_time = row['end']

        episode_df = filter_by_timestamp(dataset_df, start_time, end_time)

        print('An episode')
        print(episode_df)

        if not episode_df.empty:
            episodes.append(episode_df)

    return episodes

def save_annotated_episodes(episodes: List[pd.DataFrame], output_path: Path) -> None:
    """
    Saves each annotated episode in the list to a separate CSV file.

    Args:
        episodes (List[pd.DataFrame]): A list of episode DataFrames.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    for i, episode in enumerate(episodes):
        cls = str(episode['annotation'].iloc[0])
        cls = cls.replace(' ', '_')
        filename = f'annotated_ep_{i + 1}_{cls}.csv'

        save_pandas_dataframe_to_csv(episode, output_path / filename)
        print(f'Saved {filename}')


if __name__ == '__main__':
    # Paths
    annotated_episodes_path = get_annotations_file_path()
    input_file_path = 'synchronized_merged_selected_annotated_filtered.csv'
    input_dataset_path = get_input_path() / input_file_path
    output_path = get_output_path()

    # Read datasets
    dataset_df = read_csv_to_pandas_dataframe(input_dataset_path)
    annotations_df = read_csv_to_pandas_dataframe(annotated_episodes_path)

    # Split into annotated episodes
    annotated_episodes = split_into_annotated_episodes(dataset_df, annotations_df)

    # Save episodes
    save_annotated_episodes(annotated_episodes, output_path)