import dotenv
import os
import anthropic
from openai import OpenAI

dotenv.load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def create_embedding(text):
    return openai_client.embeddings.create(model="text-embedding-3-small", input=text)


def ask_claude(prompt):
    return anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0,
        system="You are a world-class poet. Respond only with short poems.",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )
