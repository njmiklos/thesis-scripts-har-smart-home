import requests
from typing import Dict

from get_env import get_chat_ac_info


def get_api_config(model: str = 'meta-llama-3.1-8b-instruct') -> Dict[str, str]:
    """
    Retrieves the API config details from env file and saves all information necessary 
    to make requests to the API into a dictionary. 

    Args:
        model (str, optional): The model to use for chat completions. Defaults to 'meta-llama-3.1-8b-instruct'.

    Returns:
        Dict[str, str]: A dictionary containing API key, base URL, and model name.
    """
    chat_ac_api_key, chat_ac_endpoint = get_chat_ac_info()

    api_config = {
        'api_key' : chat_ac_api_key,
        'base_url' : chat_ac_endpoint,
        'model' : model
    }

    return api_config

def send_chat_request(api_config: Dict[str, str], prompt: str, user_input: str) -> requests.Response:
    """
    Sends a request to the chat API.

    Args:
        api_config (Dict[str, str]): The API configuration containing the API key, base URL, and model name.
        prompt (str): The system prompt to set the behavior of the assistant.
        user_input (str): The user's query.

    Returns:
        Response: The HTTP response object from the API request.
    """
    headers = {
        'Authorization': f'Bearer {api_config["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': api_config['model'],
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_input}
        ]
    }
    
    chat_response = requests.post(
        url=f'{api_config["base_url"]}/chat/completions',
        headers=headers,
        json=data
    )
    
    return chat_response

def print_headers(chat_response: requests.Response) -> None:
    """
    Prints the headers of the API response.

    Args:
        chat_response (Response): The HTTP response object.
    """
    if chat_response.headers.items():
        print('Response Headers:')
        for key, value in chat_response.headers.items():
            print(f'- {key}: {value}')
    
    else:
        print('Response headers empty.')

def process_response(chat_response: requests.Response) -> None:
    """
    Processes the API response by checking its status and printing relevant details.

    Args:
        chat_response (Response): The HTTP response object.

    Raises:
        ValueError: If the response is not OK (non-2xx status code).
    """
    if chat_response.ok:
        if chat_response.status_code == 200:
            print_headers(chat_response)
            print(f'Latency: {chat_response.elapsed}')
            print(chat_response.json())
        else:
            print(f'{chat_response.status_code} {chat_response.reason}')
            print_headers(chat_response)
            print(chat_response.json())
    else:
        raise ValueError(f'Cannot access response details, response status code {chat_response.status_code} {chat_response.reason}.')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    prompt = 'You are a helpful assistant'
    user_input = 'How tall is the Eiffel tower?'

    api_config = get_api_config(model)

    chat_response = send_chat_request(api_config=api_config, prompt=prompt, user_input=user_input)

    process_response(chat_response)