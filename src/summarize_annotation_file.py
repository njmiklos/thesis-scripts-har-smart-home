import pandas as pd
from typing import List

from get_env import get_base_path
from handle_csv import read_csv_to_pandas_dataframe
from convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime
from databank_communication.query_db import get_query_result
from handle_csv import save_pandas_dataframe_to_csv


def calc_total_recorded_time() -> int:
    """
    Calculates the total recorded time in milliseconds based on the first
    and last timestamp.

    Returns:
        int: The total recorded time in milliseconds.
    """
    recorded_time_start = 1733353200000
    recorded_time_end = 1734476399000
    recorded_time_total = recorded_time_end - recorded_time_start
    return recorded_time_total

def calc_annotated_time_total(annotations: pd. DataFrame) -> int:
    """
    Calculates the total annotated time in milliseconds by summing up activity
    durations from the annotation file.

    Returns:
        int: The total annotated time in milliseconds.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        KeyError: If the expected columns ('start', 'end') are missing in the annotation file.
    """
    annotated_time_total = 0
    for _, row in annotations.iterrows():
        annotated_start_time = row['start']
        annotated_end_time = row['end']
        episode_duration = annotated_end_time - annotated_start_time
        annotated_time_total += episode_duration

    return annotated_time_total

def convert_ms_to_min(time_ms: int) -> float:
    """
    Convert time from milliseconds to minutes.

    Parameters:
        time_ms (int): Time in milliseconds.

    Returns:
        float: Time in minutes.
    """
    return time_ms / (1000 * 60)

def convert_min_to_hrs(time_min: float) -> float:
    """
    Convert time from minutes to hours.

    Parameters:
        time_min (float): Time in minutes.

    Returns:
        float: Time in hours.
    """
    return time_min / 60

def calc_percentage(recorded_time_total: int, annotated_time_total: int) -> float:
    """
    Calculates the percentage of annotated time relative to recorded time.

    Args:
        recorded_time_total: The total recorded time in milliseconds.
        annotated_time_total: The total annotated time in milliseconds.

    Returns:
        float: The percentage of annotated time relative to the total recorded time.
    """
    return (annotated_time_total * 100) / recorded_time_total

def get_annotations_df() -> pd.DataFrame:
    """
    Retrieves the annotations dataframe from the annotations file.

    Returns:
        pd.DataFrame: A dataframe containing annotations.
    """
    base_path = get_base_path()
    annotation_file = 'annotations_combined.csv'
    annotations_df = read_csv_to_pandas_dataframe(base_path / annotation_file)
    return annotations_df

def get_classes(df: pd.DataFrame) -> List[str]:
    """
    Extracts unique annotation classes from the dataframe.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        List[str]: A list of unique annotation classes.
    """
    classes = None
    classes = df['annotation'].unique()
    print(f'Classes ({len(classes)}): {classes}\n')
    return classes

def get_min_max_avg_duration_of_episodes(df: pd.DataFrame):
    """
    Determines the longest and shortest episode duration for each annotation class.
    Output them with average episode length.
    
    Args:
    df (pd.DataFrame): Input dataframe with columns 'annotation', 'start', 'end'
    
    Returns:
    None
    """
    print(f'Min. and max. episode lengths')

    for cls in df['annotation'].unique():
        subset = df[df['annotation'] == cls]
        durations = subset['end'] - subset['start']
        duration_min = convert_ms_to_min(durations.min())
        duration_max = convert_ms_to_min(durations.max())
        duration_avg = convert_ms_to_min(durations.mean())
        print(f'{cls}: min. {duration_min:.2f}, max. {duration_max:.2f}, avg. {duration_avg:.2f}')

def count_episodes(df: pd.DataFrame) -> int:
    """
    Counts the number of annotation episodes in the dataframe.

    Args:
        df (pd.DataFrame): The annotations dataframe.

    Returns:
        int: The total number of episodes.
    """
    counter = 0
    counter = len(df['annotation'])
    print(f'Number of episodes: {counter}\n')
    return counter

def calculate_duration(row: pd.Series) -> int:
    """
    Calculates the duration of an annotation episode from a dataframe row.

    Args:
        row (pd.Series): A row from the dataframe containing 'start' and 'end' columns.

    Returns:
        int: The duration in milliseconds.
    """
    return row['end'] - row['start']

def get_t_of_day(df: pd.DataFrame) -> str:
    """
    Returns the most common time of day for the annotations, based on the hour and minute of the midpoint.
    
    Args:
        df (pd.DataFrame): The annotations dataframe, with 'start' and 'end' columns representing timestamps.
    
    Returns:
        str: The most common time of day ('morning', 'afternoon', 'evening', 'night').
    """
    df = convert_timestamps_from_miliseconds_to_localized_datetime(df.copy(), 'start')
    df = convert_timestamps_from_miliseconds_to_localized_datetime(df.copy(), 'end')

    times_of_day = []

    for _, row in df.iterrows():
        midpoint = row['start'] + (row['end'] - row['start']) / 2
        hour, minute = midpoint.hour, midpoint.minute

        if (hour == 5 and minute >= 29) or (6 <= hour <= 11) or (hour == 11 and minute <= 30):
            times_of_day.append('morning')
        elif (hour == 11 and minute >= 31) or (12 <= hour <= 17) or (hour == 17 and minute <= 29):
            times_of_day.append('afternoon')
        elif (hour == 17 and minute >= 30) or (18 <= hour <= 23) or (hour == 23 and minute <= 29):
            times_of_day.append('evening')
        else:
            times_of_day.append('night')
    
    most_common_time = max(set(times_of_day), key=times_of_day.count)
    return most_common_time

def get_avg_and_std_dev_of_measurment(episodes: pd.DataFrame, measurement: str, col: str, device_no: str):
    """
    Calculate the mean and standard deviation of a specific measurement over specified time intervals.

    Args:
        episodes (pd.DataFrame): DataFrame with start and end times for querying.
        measurement (str): The measurement type (e.g., 'accelerometer').
        col (str): The column to compute statistics for (e.g., 'x').
        device_no (str): Device number to filter the data.

    Returns:
        Tuple[float, float]: Mean and standard deviation of the specified measurement, or (0, 0) if no data exists.
    """
    combined_data = []

    for _, row in episodes.iterrows():
        query = f"""
        SELECT "time", "{col}" FROM "{measurement}" WHERE "deviceNo" = '{device_no}' AND time >= {row['start']}ms AND time <= {row['end']}ms;
        """
        df = get_query_result(query)

        if not df.empty:
            combined_data.append(df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return combined_df[col].mean(), combined_df[col].std()
    else:
        return 0, 0

def get_class_information(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    """
    Returns occurrences, total time, average time, means and standard deviations of features of the annotation classes.
    Time is printed in minutes.

    Args:
        df (pd.DataFrame): The annotations dataframe.
        classes (List[str]): A list of annotation classes.
    """
    results = []
    classes_no = len(classes)
    i = 1

    for cls in classes:
        print(f'Processing class {i}/{classes_no} ({cls})')
        i = i + 1

        occurrences = (df['annotation'] == cls).sum()

        class_episodes = df[df['annotation'] == cls]
        
        durations = class_episodes.apply(calculate_duration, axis=1)
        t_total = durations.sum()
        t_total = convert_ms_to_min(t_total)
        
        t_avg = 0
        if occurrences > 0:
            t_avg = t_total / occurrences
        
        t_of_day = get_t_of_day(class_episodes)

        mean_co2, std_dev_co2 = get_avg_and_std_dev_of_measurment(class_episodes, 'co2', 'co2', '4')
        mean_air_q, std_dev_air_q = get_avg_and_std_dev_of_measurment(class_episodes, 'airquality', 'iaq', '4')
        mean_sound_min, std_dev_sound_min = get_avg_and_std_dev_of_measurment(class_episodes, 'audio', 'splMin', '5')
        mean_sound_max, std_dev_sound_max = get_avg_and_std_dev_of_measurment(class_episodes, 'audio', 'splMax', '5')

        mean_temp_entrance, std_dev_temp_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'thermometer', 'ambientTemperature', '2')
        mean_humidity_entrance, std_dev_humidity_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'hygrometer', 'humidity', '2')
        mean_light_entrance, std_dev_light_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'luxmeter', 'illuminance', '2')
        mean_acc_entrance, std_dev_acc_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'accelerometer', 'z', '2')
        mean_mag_entrance, std_dev_mag_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'magnetometer', 'z', '2')
        mean_gyr_entrance, std_dev_gyr_entrance = get_avg_and_std_dev_of_measurment(class_episodes, 'gyrometer', 'x', '2')

        mean_temp_kitchen, std_dev_temp_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'thermometer', 'ambientTemperature', '1')
        mean_humidity_kitchen, std_dev_humidity_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'hygrometer', 'humidity', '1')
        mean_light_kitchen, std_dev_light_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'luxmeter', 'illuminance', '1')
        mean_acc_kitchen, std_dev_acc_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'accelerometer', 'z', '1')
        mean_mag_kitchen, std_dev_mag_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'magnetometer', 'z', '1')
        mean_gyr_kitchen, std_dev_gyr_kitchen = get_avg_and_std_dev_of_measurment(class_episodes, 'gyrometer', 'x', '1')

        mean_temp_living_r, std_dev_temp_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'thermometer', 'ambientTemperature', '3')
        mean_humidity_living_r, std_dev_humidity_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'hygrometer', 'humidity', '3')
        mean_light_living_r, std_dev_light_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'luxmeter', 'illuminance', '3')
        mean_acc_living_r, std_dev_acc_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'accelerometer', 'z', '3')
        mean_mag_living_r, std_dev_mag_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'magnetometer', 'z', '3')
        mean_gyr_living_r, std_dev_gyr_living_r = get_avg_and_std_dev_of_measurment(class_episodes, 'gyrometer', 'x', '3')

        results.append({
            'class': cls,
            'occurrences': occurrences,
            't_total': t_total,
            't_avg': t_avg,
            't_of_day': t_of_day,
            'mean_co2': mean_co2,
            'std_dev_co2': std_dev_co2,
            'mean_air_q': mean_air_q,
            'std_dev_air_q': std_dev_air_q,
            'mean_sound_min': mean_sound_min,
            'std_dev_sound_min': std_dev_sound_min,
            'mean_sound_max': mean_sound_max,
            'std_dev_sound_max': std_dev_sound_max,
            'mean_temp_entrance': mean_temp_entrance,
            'std_dev_temp_entrance': std_dev_temp_entrance,
            'mean_humidity_entrance': mean_humidity_entrance,
            'std_dev_humidity_entrance': std_dev_humidity_entrance,
            'mean_light_entrance': mean_light_entrance,
            'std_dev_light_entrance': std_dev_light_entrance,
            'mean_acc_entrance': mean_acc_entrance,
            'std_dev_acc_entrance': std_dev_acc_entrance,
            'mean_mag_entrance': mean_mag_entrance,
            'std_dev_mag_entrance': std_dev_mag_entrance,
            'mean_gyr_entrance': mean_gyr_entrance,
            'std_dev_gyr_entrance': std_dev_gyr_entrance,
            'mean_temp_kitchen': mean_temp_kitchen,
            'std_dev_temp_kitchen': std_dev_temp_kitchen,
            'mean_humidity_kitchen': mean_humidity_kitchen,
            'std_dev_humidity_kitchen': std_dev_humidity_kitchen,
            'mean_light_kitchen': mean_light_kitchen,
            'std_dev_light_kitchen': std_dev_light_kitchen,
            'mean_acc_kitchen': mean_acc_kitchen,
            'std_dev_acc_kitchen': std_dev_acc_kitchen,
            'mean_mag_kitchen': mean_mag_kitchen,
            'std_dev_mag_kitchen': std_dev_mag_kitchen,
            'mean_gyr_kitchen': mean_gyr_kitchen,
            'std_dev_gyr_kitchen': std_dev_gyr_kitchen,
            'mean_temp_living_r': mean_temp_living_r,
            'std_dev_temp_living_r': std_dev_temp_living_r,
            'mean_humidity_living_r': mean_humidity_living_r,
            'std_dev_humidity_living_r': std_dev_humidity_living_r,
            'mean_light_living_r': mean_light_living_r,
            'std_dev_light_living_r': std_dev_light_living_r,
            'mean_acc_living_r': mean_acc_living_r,
            'std_dev_acc_living_r': std_dev_acc_living_r,
            'mean_mag_living_r': mean_mag_living_r,
            'std_dev_mag_living_r': std_dev_mag_living_r,
            'mean_gyr_living_r': mean_gyr_living_r,
            'std_dev_gyr_living_r': std_dev_gyr_living_r,

        })
    return pd.DataFrame(results)
    

if __name__ == '__main__':
    base_path = get_base_path()
    annotation_file = 'annotations_combined.csv'
    annotations = read_csv_to_pandas_dataframe(base_path / annotation_file)

    recorded_time_total_ms = calc_total_recorded_time()
    recorded_time_total_min = convert_ms_to_min(recorded_time_total_ms)
    recorded_time_total_hrs = convert_min_to_hrs(recorded_time_total_min)
    
    annotated_time_total_ms = calc_annotated_time_total(annotations)
    annotated_time_total_min = convert_ms_to_min(annotated_time_total_ms)
    annotated_time_total_hrs = convert_min_to_hrs(annotated_time_total_min)
    
    percentage_annotated = calc_percentage(recorded_time_total_ms, annotated_time_total_ms)

    print(f'Total recorded time: {recorded_time_total_ms} ms, {recorded_time_total_min:.2f} min, or {recorded_time_total_hrs:.2f} hours.')
    print(f'Total annotated time: {annotated_time_total_ms} ms, {annotated_time_total_min:.2f} min, or {annotated_time_total_hrs:.2f} hours.')
    print(f'Percentage of annotated time: {percentage_annotated:.2f}% of total recorded time.')

    annotations_df = get_annotations_df()
    get_min_max_avg_duration_of_episodes(annotations_df)
    classes = get_classes(annotations_df)
    episodes_counter = count_episodes(annotations_df)

    class_info = get_class_information(annotations_df, classes)
    save_pandas_dataframe_to_csv(class_info, base_path / 'class_info.csv')


