import dotenv
import os
import anthropic
from openai import OpenAI
import numpy as np

dotenv.load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def create_embedding(text):
    return np.array(
        openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        .data[0]
        .embedding
    )


def ask_claude(messages):
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": "user" if msg["isUser"] else "assistant",
            "content": [{"type": "text", "text": msg["content"]}]
        })
    
    return anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful AI assistant. Provide clear and concise responses.",
        messages=formatted_messages
    )
