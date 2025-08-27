# This code will read pdf files and convert them to images.
# The images will be sent to GPT

import pymupdf
import base64
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import json



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

###### MAIN ########

# Read the pdf and convert all pages as images. This is stored in contracts dir.
# The pdf is password protected
file_path = 'satyaki_hdfc_contract_password.pdf'
password = 'SAT0808'

doc = pymupdf.open(file_path,)
print("no of pages:",doc.page_count)

if doc.authenticate(password):  
    for page in doc:
        pix = page.get_pixmap()  # render page to an image
        pix.save("contracts/"+"page-%i.png" % page.number)  # store image as a PNG

doc.close()


# Now just send 1 page i.e. contracts/ page-0.png
page_0_base64 = open_image_as_base64("contracts/page-0.png")



## Let's create a message to send a text and image to the model
messages = [
    {"role": "system", "content": "You are a an expert in reading trades contract document."},
    {"role": "user", "content": [
        {"type": "text", "text": "Give me a list of information on the trades in form of list of json objects?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"{page_0_base64}"
            }
        }               
        ]}
    ]

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages
)   

contract_response = response.choices[0].message.content
with open("contract_response.json", 'w') as fp:
    fp.write(contract_response)

print(contract_response)

with open('contracts_response.json', 'r') as file:
        jdata = json.load(file)


df = pd.DataFrame(jdata)
print(df)


