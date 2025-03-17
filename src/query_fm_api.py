import requests

from get_env import get_chat_ac_info


def configure_api(model: str = 'meta-llama-3.1-8b-instruct') -> dict:
    chat_ac_api_key, chat_ac_endpoint = get_chat_ac_info()

    api_configuration = {
        'api_key' : chat_ac_api_key,
        'base_url' : chat_ac_endpoint,
        'model' : model
    }

    return api_configuration

def submit_query(api_configuration: dict, prompt: str, query: str):
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

def print_headers(response):
    print('Response Headers:')
    for key, value in response.headers.items():
        print(f'- {key}: {value}')


if __name__ == '__main__':
    model = 'meta-llama-3.1-8b-instruct'
    prompt = 'You are a helpful assistant'
    query = 'How tall is the Eiffel tower?'

    api_configuration = configure_api(model)

    response = submit_query(api_configuration=api_configuration, prompt=prompt, query=query)

    print_headers(response)
    print(response.json())