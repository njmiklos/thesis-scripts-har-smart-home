'''
This code is used to communicate with a SAIA API, and is based on tips on the product: 
https://docs.hpc.gwdg.de/services/saia/index.html
'''
from openai import OpenAI

from get_env import get_chat_ac_info

# API configuration
chat_ac_api_key, chat_ac_endpoint = get_chat_ac_info()
api_key = chat_ac_api_key
base_url = chat_ac_endpoint
model = 'meta-llama-3.1-8b-instruct'

# Start OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)

# Get response
prompt = 'You are a helpful assistant'
query = 'How tall is the Eiffel tower?'

chat_completion = client.chat.completions.create(
         messages=[{"role":"system","content":prompt},{"role":"user","content":query}],
         model= model,
     )

# Print full response as JSON
print(chat_completion) # You can extract the response text from the JSON object