"""
This module provides utilities for interacting with a chat-based API that supports both text and image input.
It is designed to streamline communication with online foundation models by handling payload construction,
image encoding, error handling, retries, and rate-limit tracking and management.

Environment Configuration:
- Ensure that your API key and endpoint are specified in the `.env` file. 
- Refer to `README.md` for full setup instructions.
"""
import requests
import base64       # Used to encode images in base64 format for API transmission
import mimetypes    # Used to detect the MIME type of an image based on its file extension
import time

from typing import Dict, Optional, Any

from utils.get_env import get_chat_ac_info


def get_api_config(model: str) -> Dict[str, str]:
    """
    Retrieves the API config details from env file and saves all information necessary 
    to make requests to the API into a dictionary. 

    Args:
        model (str): The model to use for chat completions. Some options (08.05.2025): 'internvl2.5-8b', 
            'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 'llama-4-scout-17b-16e-instruct', 
            'gemma-3-27b-it'.

    Returns:
        Dict[str, str]: A dictionary containing API key, base URL, and model name.
    """
    chat_ac_api_key, chat_ac_endpoint = get_chat_ac_info()
    return {
        'api_key' : chat_ac_api_key,
        'base_url' : chat_ac_endpoint,
        'model' : model
    }

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

def build_headers(api_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds the HTTP headers required for the API request.

    Args:
        api_config (Dict[str, Any]): Configuration dictionary containing the API key.

    Returns:
        Dict[str, str]: A dictionary of HTTP headers, including authorization and content type.
    """
    return {
        'Authorization': f'Bearer {api_config["api_key"]}',
        'Content-Type': 'application/json'
    }

def build_user_content(user_message: str, image_path: Optional[str]) -> Any:
    """
    Constructs the user content payload for the API, optionally including image data.

    Args:
        user_message (str): The textual message from the user.
        image_path (Optional[str]): Optional path to an image to include in the message.

    Returns:
        Any: A string if no image is provided, otherwise a list of content blocks (text and image).
    """
    if image_path:
        image_data = encode_image_to_base64(image_path)
        return [
            {'type': 'text', 'text': user_message},
            {'type': 'image_url', 'image_url': {'url': image_data}}
        ]
    return user_message

def build_payload(model: str, prompt: str, user_message: str, image_path: Optional[str]) -> Dict[str, Any]:
    """
    Assembles the full request payload including system prompt, user content, and model selection.

    Args:
        model (str): The model identifier for the chat completion.
        prompt (str): The system prompt to steer the assistant's behavior.
        user_message (str): The user's input message.
        image_path (Optional[str]): Optional path to an image to include in the message.

    Returns:
        Dict[str, Any]: The complete JSON payload ready for the API request.
    """
    user_content = build_user_content(user_message, image_path)
    return {
        'model': model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_content}
        ]
    }

def handle_rate_limit(response: requests.Response) -> None:
    """
    Checks the response headers for rate limit status and sleeps if nearing exhaustion.

    Args:
        response (requests.Response): The HTTP response object containing rate limit headers.

    Returns:
        None
    """
    if 'RateLimit-Remaining' in response.headers and 'RateLimit-Reset' in response.headers:
        remaining = int(response.headers['RateLimit-Remaining'])
        reset = int(response.headers['RateLimit-Reset'])
        if remaining <= 2:
            print(f'Approaching rate limit. Sleeping for {reset + 1} seconds...')
            time.sleep(reset + 1)

def handle_http_error(e: requests.exceptions.HTTPError, attempt: int, max_retries: int, backoff_factor: float) -> bool:
    """
    Handles HTTP errors and decides whether to retry.

    Returns:
        bool: True if the operation should be retried, False otherwise.
    """
    status = e.response.status_code

    if status == 429:
        if attempt > max_retries:
            raise RuntimeError('Rate limit exceeded after retries.') from e
        wait = backoff_factor * (2 ** (attempt - 1))
        print(f'WARNING. Error 429 (rate limit). Retrying in {wait}s...')
        time.sleep(wait)
        return True

    if 500 <= status < 600:
        if attempt > max_retries:
            raise RuntimeError(f'Server error {status} after retries') from e
        wait = backoff_factor * (2 ** (attempt - 1))
        print(f'WARNING. Server error {status}. Retrying in {wait}s...')
        time.sleep(wait)
        return True

    raise RuntimeError(f'HTTP error {status}: {e.response.text}') from e

def send_chat_request(model: str, timeout: Optional[int], prompt: str, user_message: str, image_path: Optional[str] = None,
                        max_retries: int = 4, backoff_factor: float = 1.0) -> requests.Response:
    """
    Sends a chat request to the API with optional image input.

    Args:
        model (str): The model to use for chat completions. Some options (08.05.2025): 'meta-llama-3.1-8b-instruct',
            'internvl2.5-8b', 'deepseek-r1-distill-llama-70b', 'deepseek-r1', 'llama-3.3-70b-instruct', 
            'llama-4-scout-17b-16e-instruct', 'gemma-3-27b-it'.
        timeout (Optional[int]): The number of seconds to wait for establishing a connection (should slightly larger 
            than a multiple of 3). If None is passed, it will wait indefinietly.
        prompt (str): The system prompt to set the behavior of the assistant.
        user_message (str): The user's query.
        image_path (Optional[str]): Optional image file path to include.
        max_retries (int): Number of retry attempts.
        backoff_factor (float): Base multiplier for exponential back-off.

    Returns:
        requests.Response: The HTTP response object from the API request.
    """
    api_config = get_api_config(model)
    headers = build_headers(api_config)
    payload = build_payload(model, prompt, user_message, image_path)
    
    attempt = 0
    while True:
        try:
            response = requests.post(f'{api_config["base_url"]}/chat/completions', headers=headers, 
                                     json=payload, timeout=timeout)
            response.raise_for_status()
            handle_rate_limit(response)
            return response

        # Request reached the server, and the server responded but with an error code
        except requests.exceptions.HTTPError as e:
            attempt += 1
            if handle_http_error(e, attempt, max_retries, backoff_factor):
                continue

        # Failed to establish connection to the server
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError('Network failure after retries') from e
            wait = backoff_factor * (2 ** (attempt - 1))
            print(f'WARNING. Network error. Retrying in {wait}s...')
            time.sleep(wait)
            continue

        # Anything else
        except requests.exceptions.RequestException as e:
            raise RuntimeError('Unexpected error communicating with API') from e

def get_headers(system_response: requests.Response) -> Dict[str, int]:
    """
    Extracts the headers of the API response.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
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

def get_rate_limits(system_response: requests.Response) -> Optional[Dict[str, int]]:
    """
    Extracts total and remaining rate limits of requests and tokens from response headers.

    Args:
        system_response (requests.Response): The HTTP response object.

    Returns:
        Optional[Dict[str, int]]: A dictionary containing token limits and remaining counts.
            If they do not exist, returns None.
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

    if rate_limits is not None:
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
        str: The latency of the response in seconds.
    """
    return str(system_response.elapsed.total_seconds())

def print_formatted_exchange(user_message: str, system_response: requests.Response) -> None:
    """
    Prints the formatted exchange and its relevant details.

    Args:
        user_message (str): User's input, i.e., message to the chatbot.
        system_response (requests.Response): The HTTP response object.
    """
    print(f'User:\n- {user_message}')
    print(f'System:\n- {get_system_message(system_response)}' )
    print(f'Latency: {get_latency(system_response)}')
    print_usage(system_response)
    print(f'JSON Response:\n{system_response.json()}')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    timeout = None
    image_path = None
    prompt = '''You are a helpful companion.'''
    user_message = '''Tell me a joke.'''

    system_response = send_chat_request(model=model, timeout=timeout, prompt=prompt, user_message=user_message, 
                                        image_path=image_path)
    print_formatted_exchange(user_message, system_response)