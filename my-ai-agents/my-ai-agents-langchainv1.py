# This code uses LangChain to create an AI agent that can provide weather information for a specified city.
# Refer to structured function calling my-structured-function/my_function_calling.py as this is detailed implementation langchainv1 framework.


import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from rich import print
from typing import Annotated
from pydantic import Field
from langchain.tools import tool
import logging
from rich.logging import RichHandler    

logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("lang_triage")


load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
    )

#-------------------- Start of the main code --------------------#

@tool
def get_weather(city: Annotated[str, Field(description="City name, spelled out fully")],) -> dict:
    """Returns weather data for a given city, a dictionary with temperature and description."""
    logger.info("Tool: get_weather invoked")
    return {"temperature": 60,"description": "Rainy",}


@tool
def get_menu() -> list:
    """Returns list of menu items for a given city."""
    logger.info("Tool: get_menu invoked")
    return ["pasta", "tomato sauce", "bell peppers", "olive oil"]
    
# Step 1: Create the agent and pass the tools
agent = create_agent(model=model, system_prompt="You're an informational agent. Answer questions cheerfully.", tools=[get_weather,get_menu])


def main():
    response = agent.invoke({"messages": [{"role": "user", "content": "Whats weather today in San Francisco and what to eat?"}]})
    print(response)
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
