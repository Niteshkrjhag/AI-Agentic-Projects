import os
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

ollama_api = os.getenv("OLLAMA_API_KEY")
print(ollama_api)

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + ollama_api}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)