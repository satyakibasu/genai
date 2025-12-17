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


# Let's define the tools for function calling
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
}]

messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What is the weather today in Pune"},
    ]

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.7,
    n=1,
    messages=messages,
    tools = tools,
    tool_choice="auto")


print(f"Response from {MODEL_NAME} on {API_HOST}")
print(response)

# We will check for the function name and invoke the custom function.
# then we will append the message and the response from the function and send to LLM.
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call.function.name)
    print(tool_call.function.arguments)

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    #print(arguments)

else:
    print(response.choices[0].message.content)

if function_name == 'lookup_weather':
    messages.append(response.choices[0].message) # append the message
    weather_response = lookup_the_weather(**arguments)
    print(weather_response)

    
    messages.append({
            "role":"tool",
            "tool_call_id":tool_call.id,
            "content":str(weather_response)})
    
    print("sending new message:",messages)

    response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.7,
    n=1,
    messages=messages,
    tools = tools)

    print(response.choices[0].message.content)
 