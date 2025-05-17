"""
This script provides utility functions to infer sensor-related metadata including unit of measurement, 
expected sampling rate, and expected precision for a given sensor reading. While the mappings are based on our project,
you can edit them according to your needs.

Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
def infer_unit(measurement: str) -> str:
    """
    Determines the unit of the given measurement based on predefined mappings for sensor data.

    The unit is inferred based on the measurement name and known sensor specifications used in university projects.
    If no unit can be determined, an empty string is returned.

    Args:
        measurement (str): The name of the measurement for which the unit should be retrieved.

    Returns:
        str: The unit corresponding to the given measurement, or an empty string if no match is found.
    """
    unit_map = {
        'temp': '°C',
        'Temperature': '°C',
        'humidity': '%',
        'acc': 'm/s²',
        'gyro': '°/s',
        'mag': 'µT',
        'illuminance': 'Lux',
        'accuracy': 'Integer',
        'airQuality': 'Index',
        'co2': 'ppm',
        'sound': 'dB',
        'voltage': 'mV',
        'load': 'Float',
        'min': 'dB',  # rssi
        'max': 'dB',  # rssi
        'mean': 'dB'    # rssi
    }

    for key, value in unit_map.items():
        if key in measurement:
            return value

    return ''

def infer_expected_sampling_rate(measurement: str) -> float:
    """
    Determines the expected sampling rate for a given measurement based on known sensor specifications.

    The expected sampling rate is inferred using the measurement name and sensor kit details from university projects.
    If no sampling rate can be determined, a random negative number (-42) is returned.

    Args:
        measurement (str): The name of the measurement for which the sampling rate should be retrieved.

    Returns:
        float: The expected sampling rate for the given measurement, or -42 if no match is found.
    """
    sampling_rate_map = {
        'air': 3,
        'iaq': 3,
        'audio': 2,
        'sound': 2,
        'spl': 2,
        'motion': 0.1,
        'acc': 0.1,
        'gyro': 0.1,
        'mag': 0.1,
        #'x': 0.1,
        #'y': 0.1,
        #'z': 0.1,
        'temperature': 10,
        'ambientTemperature': 10,
        'objectTemperature': 10,
        'thermometer': 10,
        'humidity': 10,
        'hygrometer' : 10,
        'luxmeter': 10,
        'illuminance': 10,
        'co2': 10,
        'voltage': 15,
        'min': 15,  # rssi
        'max': 15,  # rssi
        'mean': 15,  # rssi
        'load': 6,  # system load
        #'temp': 6,  # system temp
    }

    # Check for exact matches first
    for key, value in sampling_rate_map.items():
        if key == measurement:
            return value
        
     # Then check substrings
    for key, value in sampling_rate_map.items():
        if key in measurement:
            return value

    return -42

def infer_precision(measurement: str) -> int:
    """
    Determines the expected precision (number of decimal places) for a given measurement.

    The precision is inferred from the measurement name based on known sensor kit specifications.
    If the precision cannot be determined, a random negative number (-42) is returned.

    Args:
        measurement (str): The name of the measurement for which the precision should be retrieved.

    Returns:
        int: The expected number of decimal places, or -42 if no match is found.
    """
    precision_map = {
        'temp': 13,
        'Temperature': 13,
        'humidity': 12,
        'motion': 11,
        'acc': 11,
        'gyro': 11,
        'mag': 11,
        #'x': 11,
        #'y': 11,
        #'z': 11,
        'illuminance': 2,
        'airQuality': 2,
        'accuracy': 1,
        'co2': 1,
        'sound': 1,
    }

    # Check for exact matches first
    for key, value in precision_map.items():
        if key == measurement:
            return value
        
     # Then check substrings
    for key, value in precision_map.items():
        if key in measurement:
            return value

    return -42