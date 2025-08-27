import ollama
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the environment variable for API host
load_dotenv(override=True)

MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
API_HOST = os.getenv("API_HOST", "ollama")

if API_HOST == "github":
    client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = MODEL_NAME

context = """
Investor concentration risk occurs when too many investors
in a fund come from the same group, region, or category.
This can lead to liquidity risks if they withdraw at once.
"""

messages = [
    {
        "role": "system",
        "content": "You are an asset management expert. Use the provided context. If the answer is not in the context, reply with 'I don’t know'."
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: What is investor concentration risk?"
    }
]

# Option 1: Using Ollama for local inference
print(f"Using Ollama model on local and model mistral")
response = ollama.chat(model="mistral", messages=messages) 
print(response["message"]["content"])

# Option 2: Using OpenAI for remote inference
print(f"Using GitHub models...{API_HOST} and model {MODEL_NAME}")


response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.7,
    max_tokens=150,
    top_p=0.95,
    n=1,
    stop=None,
)
print(response.choices[0].message.content)
