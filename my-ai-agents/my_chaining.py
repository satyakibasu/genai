# This code is for creating agents in chains based on user input.
# The use case is to create a calendar event
# Ref: https://www.youtube.com/watch?v=bZzyPscbtI8

import os
import openai
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel, Field
from datetime import datetime


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

# --------------------------------------------------------------------------------------------------------------
# Step 1: Define the data models for each stage. This is what we expect the LLM to return in this structure
# --------------------------------------------------------------------------------------------------------------

class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""
    description:str = Field(description="Raw decsription of the event")
    is_calendar_event:bool = Field(description="whether this text decribes a calendar event")
    confidence_score:float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Get the event details by parsing"""
    name:str
    date:str
    duration_minutes:str
    participants:list[str]

class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""
    confirmation_message: str = Field(description="Natural language confirmation message")
    calendar_link:str = Field(description="Generated calendar link if applicable")




# --------------------------------------------------------------
# Let's define all the functions
# seq: extract_event_info --> parse_event_details --> generate_confirmation
# --------------------------------------------------------------

def extract_event_info(user_input:str) -> EventDetails:
    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    messages=[{"role":"system","content": f"{date_context} Analyze if the text describes a calendar event"},
                {"role": "user", "content": user_input}
            ]

    response_1 = client.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0.7,
        n=1,
        messages=messages,
        response_format=EventExtraction
    )

    print(f"Response from {MODEL_NAME} on {API_HOST}")
    #print(response_1)
    if response_1.choices[0].message.refusal:
        print(response_1.choices[0].message.refusal)
    else:
        parsed_EventExtraction = response_1.choices[0].message.parsed
        print("parsed_EventExtraction:",parsed_EventExtraction)

    return parsed_EventExtraction

def parse_event_details(event_desc) -> EventDetails:
    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    messages=[
            {"role": "system","content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference."},
            {"role": "user", "content": event_desc},
        ]
    
    response_2 = client.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0.7,
        n=1,
        messages=messages,
        response_format=EventDetails
    )
    print(f"Response from {MODEL_NAME} on {API_HOST}")
    #print(response_2)
    if response_2.choices[0].message.refusal:
        print(response_2.choices[0].message.refusal)
    else:
        parsed_EventDetails = response_2.choices[0].message.parsed
        print("parsed_EventDetails:",parsed_EventDetails)

    return parsed_EventDetails


def generate_confirmation(event_details) -> EventConfirmation:
    messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign of with your name; Satyaki",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ]

    response_3 = client.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0.7,
        n=1,
        messages=messages,
        response_format=EventConfirmation
    )
    print(f"Response from {MODEL_NAME} on {API_HOST}")
    #print(response_3)
    if response_3.choices[0].message.refusal:
        print(response_3.choices[0].message.refusal)
    else:
        parsed_EventConfirmation = response_3.choices[0].message.parsed
        print("parsed_EventConfirmation:",parsed_EventConfirmation)

    return parsed_EventConfirmation



if __name__ == '__main__':

    user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

    # --------------------------------------------------------------
    # Step 2: Let's send the 1st LLM message - Event Extraction
    # --------------------------------------------------------------
    event_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (event_extraction.confidence_score < 0.7 or not event_extraction.is_calendar_event):
        print("Gate check failed. Exiting.....")
        exit(1)
    else:
        # --------------------------------------------------------------
        # Step 3: Let's send the 2nd LLM message - Event Details
        # --------------------------------------------------------------
        event_desc = event_extraction.description
        event_detail = parse_event_details(event_desc)

    
        # --------------------------------------------------------------
        # Step 4: Let's send the 3rd LLM message - Generate Confirmation
        # --------------------------------------------------------------
        event_confirmation = generate_confirmation(event_detail)
        if event_confirmation:
            print(event_confirmation.confirmation_message)
            print(event_confirmation.calendar_link)
      


