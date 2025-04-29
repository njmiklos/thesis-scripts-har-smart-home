from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Tuple


def get_base_path() -> Path:
    """
    Fetches the directory path 'BASE_PATH' of the entire module from the .env file.

    Raises a ValueError if it is not defined in the .env file.

    Returns:
        Path: The base path.
    """
    load_dotenv()
    path = os.getenv('BASE_PATH')
    if not path:
        raise ValueError('BASE_PATH is not set in the .env file.')
    return Path(path)

def get_logging_path() -> Path:
    """
    Fetches the logging path 'LOGGING_PATH' from the .env file.
    Raises a ValueError if it is not defined in the .env file.

    Returns:
        Path: The logging path.
    """
    load_dotenv()
    path = os.getenv('LOGGING_PATH')
    if not path:
        raise ValueError('LOGGING_PATH is not set in the .env file.')
    return Path(path)

def get_output_path() -> Path:
    """
    Fetches the output path 'OUTPUTS_PATH' from the .env file.
    Raises a ValueError if it is not defined in the .env file.

    Returns:
        Path: The output path.
    """
    load_dotenv()
    path = os.getenv('OUTPUTS_PATH')
    if not path:
        raise ValueError('OUTPUTS_PATH is not set in the .env file.')
    return Path(path)

def get_input_path() -> Path:
    """
    Fetches the data path 'INPUTS_PATH' from the .env file.
    Raises a ValueError if it is not defined in the .env file.

    Returns:
        Path: The data path.
    """
    load_dotenv()
    path = os.getenv('INPUTS_PATH')
    if not path:
        raise ValueError('INPUTS_PATH is not set in the .env file.')
    return Path(path)

def get_annotations_file_path() -> Path:
    """
    Fetches the data path 'ANNOTATIONS_FILE_PATH' from the .env file.
    Raises a ValueError if it is not defined in the .env file.

    Returns:
        Path: The file path.
    """
    load_dotenv()
    path = os.getenv('ANNOTATIONS_FILE_PATH')
    if not path:
        raise ValueError('ANNOTATIONS_FILE_PATH is not set in the .env file.')
    return Path(path)

def get_database_info() -> Tuple[str, int, str]:
    """
    Fetches the variables necessary for connecting to the database.

    Loads environment variables using `dotenv` and retrieves the `HOST`, `PORT`,
    and `DATABASE_NAME`. Raises a ValueError if any of these are not defined 
    in the .env file.

    Returns:
        Tuple[str, int, str]: A tuple containing the database host, port (as an integer), 
        and database name.
    """
    load_dotenv()
    host = os.getenv('HOST')
    if not host:
        raise ValueError('HOST is not set in the .env file.')
    
    port = os.getenv('PORT')
    if not port:
        raise ValueError('PORT is not set in the .env file.')
    
    name = os.getenv('DATABASE_NAME')
    if not name:
        raise ValueError('DATABASE_NAME is not set in the .env file.')
    
    return (host, port, name)

def get_chat_ac_info() -> Tuple[str, str]:
    """
    Fetches the variables necessary for connecting to the API of the online FM.

    Loads environment variables using `dotenv` and retrieves the `CHAT_AC_API_KEY`
    and `CHAT_AC_ENDPOINT`. Raises a ValueError if any of these are not defined 
    in the .env file.

    Returns:
        Tuple[str, str]: A tuple containing the API key and the address of the endpoint.
    """
    load_dotenv()

    chat_ac_api_key = os.getenv('CHAT_AC_API_KEY')
    if not chat_ac_api_key:
        raise ValueError('CHAT_AC_API_KEY is not set in the .env file.')

    chat_ac_endpoint = os.getenv('CHAT_AC_ENDPOINT')
    if not chat_ac_endpoint:
        raise ValueError('CHAT_AC_ENDPOINT is not set in the .env file.')
    
    return (chat_ac_api_key, chat_ac_endpoint)