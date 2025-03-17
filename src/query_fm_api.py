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

def send_chat_request(api_config: Dict[str, str], prompt: str, user_message: str) -> requests.Response:
    """
    Sends a request to the chat API.

    Args:
        api_config (Dict[str, str]): The API configuration containing the API key, base URL, and model name.
        prompt (str): The system prompt to set the behavior of the assistant.
        user_message (str): The user's query.

    Returns:
        requests.Response: The HTTP response object from the API request.
    """
    headers = {
        'Authorization': f'Bearer {api_config["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': api_config['model'],
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_message}
        ]
    }
    
    system_response = requests.post(
        url=f'{api_config["base_url"]}/chat/completions',
        headers=headers,
        json=data
    )
    
    return system_response

def print_headers(system_response: requests.Response) -> None:
    """
    Prints the headers of the API response.

    Args:
        system_response (requests.Response): The HTTP response object.
    """
    if system_response.headers.items():
        print('Response Headers:')
        for key, value in system_response.headers.items():
            print(f'- {key}: {value}')
    else:
        print('Response headers empty.')

def get_request_token_usage(system_response: requests.Response) -> Dict[str, int]:
    """
    Extracts token usage details of the last exchange.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        Dict[str, int]: A dictionary containing token usage details.
    """
    exchange_tokens = system_response.json()['usage']
    return exchange_tokens

def get_rate_limits(system_response: requests.Response) -> Dict[str, int]:
    """
    Extracts total and remaining rate limits of requests and tokens from response headers.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        Dict[str, int]: A dictionary containing token limits and remaining counts.
    """
    rate_limits = {}
    for key, value in system_response.headers.items():
        if 'Limit' in key:
            rate_limits[key] = int(value)
    return rate_limits

def print_usage(system_response: requests.Response) -> None:
    """
    Prints the token usage.

    Args:
        system_response (requests.Response): The HTTP response object.
    """
    response_tokens = get_request_token_usage(system_response)
    total_tokens = get_rate_limits(system_response)

    print('Tokens used for request:')
    for key, value in response_tokens.items():
        print(f'- {key}: {value}')

    print('Total rate limits:')
    for key, value in total_tokens.items():
        print(f'- {key}: {value}')
            
def get_system_message(system_response: requests.Response) -> str:
    """
    Extracts the systems's response text from the API response.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        str: The systems's response message.
    """
    system_response_text = system_response.json()['choices'][0]['message']['content']
    return system_response_text

def get_latency(system_response: requests.Response) -> str:
    """
    Extracts the API response latency.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        str: The latency of the response.
    """
    return str(system_response.elapsed)

def print_formatted_exchange(user_message: str, system_response: requests.Response) -> None:
    """
    Prints the formatted exchange and its relevant details.

    Args:
        user_message (str): User's input, i.e., message to the chatbot.
        system_response (requests.Response): The HTTP response object.

    Raises:
        ValueError: If the response is not OK (non-2xx status code).
    """
    if system_response.ok:
        if system_response.status_code != 200:
            print(f'{system_response.status_code} {system_response.reason}')

        print(f'User:\n- {user_message}')
        print(f'System:\n- {get_system_message(system_response)}' )
        print(f'Latency: {get_latency(system_response)}')
        print_usage(system_response)
    else:
        raise ValueError(f'Cannot access response details, response status code {system_response.status_code} {system_response.reason}.')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    prompt = 'You are a helpful assistant'
    user_message = 'How tall is the Eiffel tower?'

    api_config = get_api_config(model)

    system_response = send_chat_request(api_config=api_config, prompt=prompt, user_message=user_message)

    print_formatted_exchange(user_message, system_response)