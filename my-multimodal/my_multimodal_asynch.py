import os
from dotenv import load_dotenv
import openai
import base64
import asyncio


load_dotenv()

API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

if API_HOST == "github":
    asynch_client = openai.AsyncOpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])


# Convert images to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    return image_base64



async def generate_response(id, message):
    print(f"generating message for : {id}")
    response = await asynch_client.chat.completions.create(model=MODEL_NAME, messages=message,)
    print(f"Received message for : {id}")

    return id, response.choices[0].message.content
    

# USING DATA URI FOR IMAGE
## Let's create a message to send a text and image to the model
base64_image = encode_image_to_base64("pexels-photo-1054655.jpeg")
datauri_messages = [
    {"role": "system", "content": "You are a helpful assistant that helps people find information."},
    {"role": "user", "content": [
        {"type": "text", "text": "What is this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }               
        ]}
    ]


# GRAPH ANALYSIS
g_messages = [
    {
        "role": "user",
        "content": [
            {"text": "What zone are we losing the most trees in?", "type": "text"},
            {
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/20210331_Global_tree_cover_loss_-_World_Resources_Institute.svg/1280px-20210331_Global_tree_cover_loss_-_World_Resources_Institute.svg.png"
                },
                "type": "image_url",
            },
        ],
    }
]

# INSURANCE CLAIM
image_encoded = encode_image_to_base64("dented_car.jpg")
insurance_message=[
        {
            "role": "system",
            "content": (
                "You are an AI assistant that helps auto insurance companies process claims."
                "You accept images of damaged cars that are submitted with claims, and you are able to make judgments "
                "about the causes of automobile damage, and the validity of claims regarding that damage."
            ),
        },
        {
            "role": "user",
            "content": [
                {"text": "Claim states that this damage is due to hail. Is it valid?", "type": "text"},
                {"image_url": {"url": f"data:image/png;base64,{image_encoded}"}, "type": "image_url"},
            ],
        },
    ]

# TABLE ANALYSIS
table_image = encode_image_to_base64("page_0.png")
table_message=[
        {
            "role": "user",
            "content": [
                {"text": "What's the cheapest plant?", "type": "text"},
                {"image_url": {"url": f"data:image/png;base64,{table_image}"}, "type": "image_url"},
            ],
        }
    ]

# This funtion will print when all the messages have completed.
async def main():
    print("Using model: ",MODEL_NAME)
    responses = await asyncio.gather( 
    generate_response("data uri", datauri_messages),
    generate_response("graph", g_messages),
    generate_response("insurance", insurance_message),
    generate_response("table", table_message),
    )
    
    #print(responses)
    for id, response in responses:
        print(id,": \n",response,"\n")

# This function will print as n when the message is ready
async def main_ascompleted():
    all_messages = [("data uri",datauri_messages),("graph", g_messages),("insurance", insurance_message),("table", table_message)]
    tasks = [generate_response(id,msg) for id,msg in all_messages]


    for completed in asyncio.as_completed(tasks):
        id, message = await completed
        print(id,": \n",message,"\n")

#asyncio.run(main())
asyncio.run(main_ascompleted())