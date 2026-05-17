import os
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

class LocalOllamaLLM:
    def __init__(self, model = "gpt-oss:120b"):
        self.api_key = os.getenv("OLLAMA_API_KEY")
        self.client = Client(
            host = "https://ollama.com",
            headers = {'Authorization': f"Bearer {self.api_key}"}
        )
        self.model = model

    def __call__(self, prompt:str)-> str:
        messages = [
            {"role":"user", "content":prompt}
        ]
        response_text = ""
        
        for part in self.client.chat(self.model, messages=messages, stream= True):
            chunk = part["message"]["content"]
            response_text += chunk
            print(part['message']['content'], end='', flush=True)
    

if __name__ == '__main__':
    model = LocalOllamaLLM()
    print(model("Hello, How are you ?"))