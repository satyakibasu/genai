import os

#import azure.identity
import openai
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure, OpenAI.com, or Ollama API
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    print("Using GitHub models...")
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
elif API_HOST == "ollama":
    print("Using Ollama model on local...")
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")

else:
    raise ValueError(f"Unsupported API_HOST: {API_HOST}")

responses = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.7,
    n=1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that makes lots of cat references and uses emojis."},
        {"role": "user", "content": "Write a single line about a hungry cat who wants tuna"},
    ],
    stream=True,  # Enable streaming for the response
)
print(f"Response from {API_HOST}: using model: {MODEL_NAME}\n")
#print(responses.choices[0].message.content) #--> enable this line to print the full response at once


# Print the response as it streams in
for event in responses:
    
    if event.choices:
        content = event.choices[0].delta.content
        if content:
            print(content, end="", flush=True)  # Print the content as it arrives
print("\n\n")  # Add a newline at the end for better readability
print("End of response.\n")


#Chat history + few shot examples
messages = [
    {"role": "system", "content": "I am a teaching assistant helping with Python questions"},
    {"role": "user", "content": "What is the difference between a list and a tuple in Python?"},
    {"role": "assistant", "content": "A list is mutable, meaning you can change its content, while a tuple is immutable, meaning once created, it cannot be changed."},
    {"role": "user", "content": "Can you give an example of how to create a dictionary in Python?"},
    {"role": "assistant", "content": "Sure! You can create a dictionary using curly braces: `my_dict = {'key1': 'value1', 'key2': 'value2'}`."},
    ]

while True:
    question = input("\nYour question (type exit/quit/q to exit): ")
    if question.lower() in ["exit", "quit", "q"]:
        print("Exiting the chat. Goodbye!")
        break
    if not question.strip():
        print("Please enter a valid question.")
        continue

    print("Sending question...")

    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=1,
        max_tokens=400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    bot_response = response.choices[0].message.content
    #messages.append({"role": "assistant", "content": bot_response})

    print("Answer: ")
    print(bot_response)

    # Let's do chained calls to summarize the conversation
    chain_message = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "assistant", "content": "You will summarize the conversation in a concise manner."},
            {"role": "user", "content": "Summarize the following conversation:\n "+bot_response}]
            

    print("\n\nNow summarizing the conversation...")
    summary_response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.5,
        max_tokens=100,
        top_p=0.95, 
        messages=chain_message, 
    )

    summary = summary_response.choices[0].message.content
    print("Summary of the conversation: ")
    print(summary)
