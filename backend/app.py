from dotenv import load_dotenv
import os
import anthropic
from openai import OpenAI
import numpy as np
import logging
import sys

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

# Load embeddings
try:
    embeddings = np.load("embeddings.npy")
    with open("texts.txt", "r") as f:
        texts = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    embeddings = np.array([])
    texts = []


def ask_rag(messages):
    """
    messages : list[dict]
        list of messages.
            each message is a dict with keys "role" and "content".
                "role" key has a value of either "user" or "bot"
                "content" key has a value of the text of the message

    Returns
    response : dict
       single key "content" containing the response from the model.
    """

    try:
        logger.info(f"Processing query with {len(messages)} messages")
        query = messages[-1]["content"]
        logger.info(f"Creating embedding for query: {query[:100]}...")

        query_embedding = create_embedding(query)
        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_context = [texts[i] for i in top_indices]

        logger.info(f"found {len(relevant_context)} relevant contexts")
        logger.debug(f"top similarity scores: {[similarities[i] for i in top_indices]}")

        context_prompt = (
            f"\n\nRelevant Context:\n"
            + "\n".join(relevant_context)
            + f"\n\nQ: {query}\nA:"
        )

        messages_with_context = messages[:-1] + [
            {"role": "user", "content": messages[-1]["content"] + context_prompt}
        ]

        logger.info("sending request to Claude")
        response = ask_claude(messages_with_context)
        logger.info("received response from Claude")
        return response

    except Exception as e:
        logger.error(f"error in ask_rag: {str(e)}", exc_info=True)
        raise


def create_embedding(text):
    return np.array(
        openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        .data[0]
        .embedding
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def sort_by_similarity(query_embedding, embeddings):
    return sorted(
        embeddings,
        key=lambda x: cosine_similarity(x, query_embedding),
        reverse=True,
    )


def ask_claude(messages):
    try:
        logger.info("Formatting messages for Claude")
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        logger.info("Sending request to Claude API")
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            system="",  # TODO add system prompt
            messages=formatted_messages,
        )
        logger.info("Received response from Claude API")
        return response
    except Exception as e:
        logger.error(f"Error in ask_claude: {str(e)}", exc_info=True)
        raise


def test_ask_rag():
    logger.info("Starting rag test")

    try:
        test_messages = [
            {
                "role": "user",
                "content": "What can I use to move components from a conveyor belt?",
            }
        ]

        logger.info("Testing ask_rag with sample query")
        response = ask_rag(test_messages)

        logger.info("Test successful!")
        logger.info(f"Response received: {response.content[0].text[:100]}...")

        with open("test_response.txt", "w") as f:
            f.write(response.content[0].text)
        logger.info("Response saved to test_response.txt")
        return response

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Running test function")
    test_ask_rag()
