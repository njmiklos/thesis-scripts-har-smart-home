import pandas as pd
from typing import List
from pathlib import Path

from get_env import get_input_path, get_output_path, get_annotations_file_path
from handle_csv import read_csv_to_pandas_dataframe, save_pandas_dataframe_to_csv
from filter_df import filter_by_timestamp


def split_into_annotated_episodes(df: pd.DataFrame, episodes_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Splits sensor data into episodes based on the list of the annotated episodes.

    Args:
        df (pd.DataFrame): The DataFrame containing the event data.
        episodes_df (pd.DataFrame): The DataFrame containing annotated episodes.

    Returns:
        List[Tuple[str, pd.DataFrame]]: List of tuples (annotation_label, episode_df).
    """
    episodes = []

    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).view('int64') // 10**6
    annotations_df['start'] = pd.to_datetime(annotations_df['start'], unit='ms', utc=True).view('int64') // 10**6
    annotations_df['end'] = pd.to_datetime(annotations_df['end'], unit='ms', utc=True).view('int64') // 10**6

    for _, row in annotations_df.iterrows():
        start_time = row['start']
        end_time = row['end']
        annotation_label = row['annotation']

        episode_df = filter_by_timestamp(df, start_time, end_time)
        if not episode_df.empty:
            episode_df['annotation'] = annotation_label
            episodes.append((annotation_label, episode_df))

    return episodes

def save_annotated_episodes(episodes: List[pd.DataFrame], output_path: Path) -> None:
    """
    Saves each annotated episode in the list to a separate CSV file with a timestamp in ms UTC Europe/Berlin.

    Args:
        episodes (List[Tuple[str, pd.DataFrame]]): A list of tuples containing the annotation and DataFrame.
        output_path (Path): The directory where the CSV files will be saved.

    Returns:
        None
    """
    for i, (annotation_label, episode) in enumerate(episodes):
        berlin_time = pd.to_datetime(episode['time'], unit='ms', utc=True).dt.tz_convert('Europe/Berlin')
        episode['time'] = berlin_time.view('int64') // 10**6

        annotation_name = str(annotation_label).replace(' ', '_')
        filename = f'annotated_ep_{i + 1}_{annotation_name}.csv'

        save_pandas_dataframe_to_csv(episode, output_path / filename)
        print(f'Saved {filename}')


if __name__ == '__main__':
    # Paths
    annotated_episodes_path = get_annotations_file_path()
    input_file_path = 'synchronized_merged_selected_annotated_filtered.csv'
    input_dataset_path = get_input_path() / input_file_path
    output_path = get_output_path()

    # Read datasets
    df = read_csv_to_pandas_dataframe(input_dataset_path)
    annotations_df = read_csv_to_pandas_dataframe(annotated_episodes_path)

    # Split into annotated episodes
    annotated_episodes = split_into_annotated_episodes(df, annotations_df)

    # Save episodes
    save_annotated_episodes(annotated_episodes, output_path)