import os
import openai
from dotenv import load_dotenv
import json
from rich import print

load_dotenv()

API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    print("Using GitHub models...")
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
elif API_HOST == "ollama":
    print("Using Ollama model on local...")
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")

def lookup_the_weather(city_name=None, zip_code=None):
    """Lookup the weather for a given city name or zip code."""
    print(f"Looking up weather for {city_name or zip_code}...")
    return {"city_name": city_name,"zip_code": zip_code,"weather":"sunny","temperature":75}


# Let's define the tools for function calling. This has multiple functions.
tools = [{
          "type": "function",
          "function":{
                "name": "lookup_weather",
                "description": "Lookup the weather for a given city name or zip code.",
                "parameters": {
                            "type": "object",
                            "properties": {
                                            "city_name": {"type": "string","description": "The city name",},
                                            "zip_code": {"type": "string","description": "The zip code",},
                                        },
                            "additionalProperties": False,
                            }
                    }  
},
        {
          "type": "function",
          "function":{
                "name": "lookup_movies",
                "description": "Lookup the movies playing in the given city or zip code",
                "parameters": {
                            "type": "object",
                            "properties": {
                                            "city_name": {"type": "string","description": "The city name",},
                                            "zip_code": {"type": "string","description": "The zip code",},
                                        },
                            "additionalProperties": False,
                            }
                    }  
},]

messages = [
        {"role": "system", "content": "You are a tourism chatbot. "},
        {"role": "user", "content": "Is is rainy in Pune and which movies to watch for?"},
    ]

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.7,
    n=1,
    messages=messages,
    tools = tools)

print(f"Response from {MODEL_NAME} on {API_HOST}")
print(response)

for res in response.choices[0].message.tool_calls:
    print(res.function.name)
    print(res.function.arguments)


# use asynchio to call these functions in parallel. Use Task Group