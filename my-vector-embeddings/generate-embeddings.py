import os
import dotenv
import openai
import json

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize OpenAI client with the base URL and API key from environment variables
openai_client = openai.OpenAI(base_url="https://models.github.ai/inference",api_key=os.environ["GITHUB_TOKEN"])

MODEL_NAME = "openai/text-embedding-3-small"
MODEL_DIMENSIONS = 1536
input_text = "hello world, this is a test embedding"

embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    dimensions=MODEL_DIMENSIONS,
    input=input_text,
)

# Print the generated embeddings
#print(embeddings_response.data[0].embedding)
print(len(embeddings_response.data[0].embedding))  # Should print 1536

# Store the embeddings in a file in a dict format with key as input_text

with open("embeddings.json", "w") as f:
    json.dump({input_text: embeddings_response.data[0].embedding}, f)