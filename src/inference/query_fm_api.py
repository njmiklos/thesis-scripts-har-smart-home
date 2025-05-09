import requests
import base64   # Used to encode images in base64 format for API transmission
import mimetypes    # Used to detect the MIME type of an image based on its file extension

from typing import Dict, Optional

from utils.get_env import get_chat_ac_info


def get_api_config(model: str) -> Dict[str, str]:
    """
    Retrieves the API config details from env file and saves all information necessary 
    to make requests to the API into a dictionary. 

    Args:
        model (str, optional): The model to use for chat completions. Some options (08.05.2025): 'internvl2.5-8b', 
            'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 'llama-4-scout-17b-16e-instruct', 
            'gemma-3-27b-it'.

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

def detect_image_type(image_path: str) -> str:
    """
    Detects the MIME type of an image based on its file extension.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The MIME type of the image (e.g., 'image/png').
             Falls back to 'application/octet-stream' if undetectable.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
    return mime_type

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file into a base64 data URI with MIME type.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded data URI (e.g., 'data:image/png;base64,...').
    """
    mime_type = detect_image_type(image_path)

    with open(image_path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')

    return f'data:{mime_type};base64,{encoded}'

def send_chat_request(model: str, prompt: str, user_message: str, image_path: Optional[str] = None) -> requests.Response:
    """
    Sends a chat request to the API with optional image input.

    Args:
        model (str]): The model to use for chat completions. Some options (08.05.2025): 'meta-llama-3.1-8b-instruct',
            'internvl2.5-8b', 'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 
            'llama-4-scout-17b-16e-instruct', 'gemma-3-27b-it'.
        prompt (str): The system prompt to set the behavior of the assistant.
        user_message (str): The user's query.
        image_path (Optional[str]): Optional image file path to include.

    Returns:
        requests.Response: The HTTP response object from the API request.
    """
    api_config = get_api_config(model)

    headers = {
        'Authorization': f'Bearer {api_config["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    if image_path:
        image_data = encode_image_to_base64(image_path)
        user_content = [
            {'type': 'text', 'text': user_message},
            {'type': 'image_url', 'image_url': {'url': image_data}}
        ]
    else:
        user_content = user_message

    payload = {
        'model': api_config['model'],
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_content}
        ]
    }
    
    system_response = requests.post(
        url=f'{api_config["base_url"]}/chat/completions',
        headers=headers,
        json=payload
    )
    
    return system_response

def get_headers(system_response: requests.Response) -> Dict[str, int]:
    """
    Extracts the headers of the API response.

    Args:
        Dict[str, int]: A dictionary containing headers.
    """
    headers = {}
    for key, value in system_response.headers.items():
        headers[key] = value
    return headers

def get_request_token_usage(system_response: requests.Response) -> Dict[str, int]:
    """
    Extracts token usage details of the last exchange.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        Dict[str, int]: A dictionary containing token usage details:
        - 'prompt_tokens': Tokens describing the size of user's input.
        - 'completion_tokens': Tokens the model needs to generate a response.
        - 'total_tokens': The sum of prompt and completion tokens.
    """
    exchange_tokens = system_response.json()['usage']
    return exchange_tokens

def get_request_total_tokens(system_response: requests.Response) -> int:
    """
    Extracts sum of prompt and completion tokens from the response.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        total_tokens (int): The sum of prompt and completion tokens.
    """
    exchange_tokens = get_request_token_usage(system_response)
    return exchange_tokens['total_tokens']

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
    headers = get_headers(system_response)
    tokens = get_request_token_usage(system_response)
    rate_limits = get_rate_limits(system_response)

    print('Headers:')
    for key, value in headers.items():
        print(f'- {key}: {value}')

    print('Tokens used for request:')
    for key, value in tokens.items():
        print(f'- {key}: {value}')

    print('Total rate limits:')
    for key, value in rate_limits.items():
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

    image_path = None

    prompt = '''You are a helpful companion.'''
    
    user_message = '''Tell me a joke.'''

    system_response = send_chat_request(model=model, prompt=prompt, user_message=user_message, image_path=image_path)

    print_formatted_exchange(user_message, system_response)