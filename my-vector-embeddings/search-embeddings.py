import json

#read the embeddings response from a file
with open('embeddings.json', 'r') as file:
    embeddings_response = json.load(file)


MODEL_DIMENSIONS = 1536 # Must match the dimension of the embeddings

import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai_client = openai.OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"]
)
MODEL_NAME = "openai/text-embedding-3-small"

# Define the function to generate embeddings
def generate_embeddings(texts):
    response = openai_client.embeddings.create(
        model=MODEL_NAME,
        input=texts
    )
    return response

