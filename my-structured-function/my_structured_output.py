import os
import openai
from dotenv import load_dotenv
import json
from rich import print
from pydantic import BaseModel, Field
from enum import Enum

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

class DayOfWeek(str, Enum):
    SUNDAY = "Sunday"
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"

class CalenderEvent(BaseModel):
    name:str
    #date:str = Field(description="date in format YYYY-MM-DD")
    date:str = DayOfWeek
    participants: list[str]



messages = [
        {"role": "system", "content": "Extract the event information. If no year mentioned then assume current year 2025"},
        #{"role": "user", "content": "Alice and Bob are doing a science fair on on 1st of August."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

response = client.chat.completions.parse(
    model=MODEL_NAME,
    temperature=0.7,
    n=1,
    messages=messages,
    response_format=CalenderEvent
)


print(f"Response from {MODEL_NAME} on {API_HOST}")
print(response)
if response.choices[0].message.refusal:
    print(response.choices[0].message.refusal)
else:
    print(response.choices[0].message.parsed)


 