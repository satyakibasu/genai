# This code will read pdf files and convert them to images.
# The images will be sent to GPT

import pymupdf
import base64
import os
from dotenv import load_dotenv
import openai
import pandas as pd
from pydantic import BaseModel 
from rich import print


# load env variable
load_dotenv()

# Get from env variables
API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

if API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])


# Send images to LLM as base64
def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:image/png;base64,{image_base64}"


def generate_pdf_images(file_name, password):
    folder_name = "contracts"
    file_path = folder_name+"/"+file_name
    name_without_ext = os.path.splitext(file_name)[0]

    with pymupdf.open(file_path) as doc:    
        page_count = doc.page_count
        print("no of pages:",doc.page_count)

        if doc.authenticate(password):  
            for page in doc:
                pix = page.get_pixmap()  # render page to an image
                pix.save(folder_name+"/images/"+name_without_ext+"-page-%i.png" % page.number)  # store image as a PNG
    
    return page_count



# Let create the message structure for the trade reconciliation task
class TradeExtractionMessage(BaseModel):
    trader_name: str
    trade_id: str
    trade_date: str
    order_time: str
    security_symbol: str
    security_name: str
    quantity: int
    price: float
    counterparty: str
    status: str
    buy_sell: str = "unknown"
    notes: str = None


class TraderDetails(BaseModel):
    trader_name: str
    trade_date: str
    total_quantity: int
    individual_quantities: list[int] = None
    identification: str = None

def extract_trader_details(image_base64: str):
    messages = [
    {"role": "system", "content": "You are a an expert in reading trades contract document."},
    {"role": "user", "content": 
        [
            {"type": "text", "text": "Give me the name of the person and the trade date who has traded information. Add the total quantity of trades executed. Do not put any header or footer string information "},
            {"type": "image_url","image_url": {"url": f"{image_base64}"}}               
        ]
    }
    ]
    
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
        response_format=TraderDetails
    )

    #print(f"Response from {MODEL_NAME} on {API_HOST}")
    trader_details = response.choices[0].message.content
    print("Trader name:",trader_details)
    
    return trader_details

# Let's define the function to call the LLM which will extract the trade details from the image
def extract_trade_details(image_base64: str)->list[TradeExtractionMessage]:
    messages = [
    {"role": "system", "content": "You are a an expert in reading trades contract document."},
    {"role": "user", "content": 
        [
            {"type": "text", "text": "Give me a list of information on the trades executed in the document. Provide in TradeExtractionMessage. Ensure there is a security symbol, name and if buy/sell. Add notes if any."},
            {"type": "image_url","image_url": {"url": f"{image_base64}"}}               
        ]
    }
    ]
    
    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
        response_format=TradeExtractionMessage
    )

    #print(f"Response from {MODEL_NAME} on {API_HOST}")
    raw_trade_extract = response.choices[0].message.content
    print("Raw trade extract:",raw_trade_extract)
    
    return raw_trade_extract


def format_trade_details(raw_trade_extract:str, trader_details:str) -> list[TradeExtractionMessage]:
    # Use the LLM to format the extracted trade details into structured JSON using the TradeExtractionMessage
    messages = [
        {"role": "system", "content": "You are a helpful assistant that formats trade details into structured JSON. Ensure there is a trader name, trade date, order time, symbol, total quantity, price, counterparty, status, buy/sell and notes if any."},
        {"role": "user", "content": f"Format the following trade details into a JSON array of TradeExtractionMessage objects: {raw_trade_extract} and trader details and total quantity from {trader_details}. Provide in TradeExtractionMessage."},
    ]   

    response = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.5,
        response_format=TradeExtractionMessage
    )    

    print(f"Response from {MODEL_NAME} on {API_HOST}")
    if response.choices[0].message.refusal:
        print(response.choices[0].message.refusal)
        return None
    else:
        parsed_trade_details = response.choices[0].message.parsed
        
        return parsed_trade_details



if __name__ == "__main__":
    # Example usage
    #pdf_file = "varun.pdf"
    #pdf_password = ""

    # Step 1: Convert PDF to images
    #num_pages = generate_pdf_images(pdf_file, pdf_password)
    #print(f"Generated images for {num_pages} pages.")


    #image_base64 = open_image_as_base64('contracts/images/satyaki-page-0.png')
    image_base64 = open_image_as_base64('contracts/images/varun-page-0.png')


    trader_details = extract_trader_details(image_base64) # agent 1
    raw_trade_details = extract_trade_details(image_base64) # agent 2
    parsed_trade_details = format_trade_details(raw_trade_details, trader_details) #agent 3

    print("Parsed trade details:",parsed_trade_details)

    #df = pd.DataFrame([parsed_trade_details.model_dump()])
    #print(df)
