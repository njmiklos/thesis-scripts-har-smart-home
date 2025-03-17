import requests
from typing import Dict, Any, Optional, Tuple

from get_env import get_chat_ac_info


def configure_api(model: str = 'meta-llama-3.1-8b-instruct') -> Dict[str, str]:
    """
    Configures the API settings by grabbing information from env file, 
    then saving all information necessary to make requests to the API into a dictionary. 

    Args:
        model (str, optional): The model to use for chat completions. Defaults to 'meta-llama-3.1-8b-instruct'.

    Returns:
        Dict[str, str]: A dictionary containing API key, base URL, and model name.
    """
    chat_ac_api_key, chat_ac_endpoint = get_chat_ac_info()

    api_configuration = {
        'api_key' : chat_ac_api_key,
        'base_url' : chat_ac_endpoint,
        'model' : model
    }

    return api_configuration

def submit_query(api_configuration: Dict[str, str], prompt: str, query: str) -> requests.Response:
    """
    Submits a query to the chat API.

    Args:
        api_configuration (Dict[str, str]): The API configuration containing the API key, base URL, and model name.
        prompt (str): The system prompt to set the behavior of the assistant.
        query (str): The user's query.

    Returns:
        Response: The HTTP response object from the API request.
    """
    headers = {
        'Authorization': f'Bearer {api_configuration["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': api_configuration['model'],
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
    }
    
    response = requests.post(
        url=f'{api_configuration["base_url"]}/chat/completions',
        headers=headers,
        json=data
    )
    
    return response

def print_headers(response: requests.Response) -> None:
    """
    Prints the headers of the API response.

    Args:
        response (Response): The HTTP response object.
    """
    if response.headers.items() is not None:
        print('Response Headers:')
        for key, value in response.headers.items():
            print(f'- {key}: {value}')
    
    else:
        print('Response headers empty.')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    prompt = 'You are a helpful assistant'
    query = 'How tall is the Eiffel tower?'

    api_configuration = configure_api(model)

    response = submit_query(api_configuration=api_configuration, prompt=prompt, query=query)

    print_headers(response)
    print(response.json())