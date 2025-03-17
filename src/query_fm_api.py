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
        chat_response (requests.Response): The HTTP response object.
    """
    if chat_response.headers.items():
        print('Response Headers:')
        for key, value in chat_response.headers.items():
            print(f'- {key}: {value}')
    else:
        print('Response headers empty.')

def get_response_token_usage(chat_response: requests.Response) -> Dict[str, int]:
    """
    Extracts token usage details from the API response.

    Args:
        chat_response (requests.Response): The HTTP response object.

    Returns:
        Dict[str, int]: A dictionary containing token usage details.
    """
    response_tokens = chat_response.json()['usage']
    return response_tokens

def get_total_token_usage(chat_response: requests.Response) -> Dict[str, int]:
    """
    Extracts token limits and remaining tokens from response headers.

    Args:
        chat_response (requests.Response): The HTTP response object.

    Returns:
        Dict[str, int]: A dictionary containing token limits and remaining counts.
    """
    total_tokens = {}
    for key, value in chat_response.headers.items():
        if 'Limit' in key:
            total_tokens[key] = int(value)
    return total_tokens

def print_usage(chat_response: requests.Response) -> None:
    """
    Prints the token usage.

    Args:
        chat_response (requests.Response): The HTTP response object.
    """
    response_tokens = get_response_token_usage(chat_response)
    total_tokens = get_total_token_usage(chat_response)

    print('Tokens used for response:')
    for key, value in response_tokens.items():
        print(f'- {key}: {value}')

    print('Total tokens:')
    for key, value in total_tokens.items():
        print(f'- {key}: {value}')
            
def get_response_txt(chat_response: requests.Response) -> str:
    """
    Extracts the assistant's response text from the API response.

    Args:
        chat_response (requests.Response): The HTTP response object.

    Returns:
        str: The assistant's response message.
    """
    response_txt = chat_response.json()['choices'][0]['message']['content']
    return response_txt

def get_latency(chat_response: requests.Response) -> str:
    """
    Extracts the API response latency.

    Args:
        chat_response (requests.Response): The HTTP response object.

    Returns:
        str: The latency of the response.
    """
    return str(chat_response.elapsed)

def print_formatted_exchange(message: str, chat_response: requests.Response) -> None:
    """
    Prints the formatted exchange and its relevant details.

    Args:
        message (str): User's input, i.e., message to the chatbot.
        chat_response (requests.Response): The HTTP response object.

    Raises:
        ValueError: If the response is not OK (non-2xx status code).
    """
    if chat_response.ok:
        if chat_response.status_code != 200:
            print(f'{chat_response.status_code} {chat_response.reason}')

        print(f'User:\n- {message}')
        print(f'Model:\n- {get_response_txt(chat_response)}' )
        print(f'Latency: {get_latency(chat_response)}')
        print_usage(chat_response)
    else:
        raise ValueError(f'Cannot access response details, response status code {chat_response.status_code} {chat_response.reason}.')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    prompt = 'You are a helpful assistant'
    message = 'How tall is the Eiffel tower?'

    api_config = get_api_config(model)

    chat_response = send_chat_request(api_config=api_config, prompt=prompt, user_message=message)

    print_formatted_exchange(message, chat_response)