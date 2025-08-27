import os
from dotenv import load_dotenv
import openai
import base64


load_dotenv()

API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

if API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])


# Convert images to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    return image_base64


# reading an image which contains text
#base64_image = encode_image_to_base64("chess_participants.jpg")
base64_image = encode_image_to_base64("contracts/page-0.png")


# DESCRIBE THE IMAGE

## Let's create a message to send a text and image to the model
messages = [
    {"role": "system", "content": "You are a helpful assistant that helps people find information."},
    {"role": "user", "content": [
        {"type": "text", "text": "Give me a list of participants?"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }               
        ]}
    ]





response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages
)   

answer_datauri = response.choices[0].message.content
print(answer_datauri)

exit(1)

#chaining the above message to find out the schedule
new_message = [
                {"role": "system", "content":"You are an chess assitant organizing a chess tournament"},
                {"role": "user", "content":f"Using the context Context:\n\n{answer_datauri}, Give me the schedule of under 10 age participants playing in round robin mode"},
            ]


response = client.chat.completions.create(model=MODEL_NAME, messages=new_message, temperature=0.5)
chess_schedule = response.choices[0].message.content

print(chess_schedule)



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
response = client.chat.completions.create(model=MODEL_NAME, messages=g_messages, temperature=0.5)
answer_graph = response.choices[0].message.content

# INSURANCE CLAIM

image_encoded = encode_image_to_base64("dented_car.jpg")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
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
    ],
)

answer_insurance = response.choices[0].message.content



# TABLE ANALYSIS

table_image = encode_image_to_base64("page_0.png")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "What's the cheapest plant?", "type": "text"},
                {"image_url": {"url": f"data:image/png;base64,{table_image}"}, "type": "image_url"},
            ],
        }
    ],
)

answer_table = response.choices[0].message.content

