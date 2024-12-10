import openai
import dotenv
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

import requests

def ask_openai(prompt):
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

print(ask_openai("Say this is a test!"))

