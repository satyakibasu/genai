import requests ##access the Ollama via the API
import json
import gradio as gr ##gradio for the web interface
from rich import print ##rich for better logging

#OLLAMA_API_URL = "http://localhost:11434/api/generate"  #--> llama API endpoint
OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions" #--> behaves like OpenAI Chat Completions
MODEL_NAME = "codeguru"  ##custom code assistant model

headers = {
    "Content-Type": "application/json"
}

chat_history = []

def generate_response(prompt):
    """Generate a response from the Ollama API based on the user prompt."""

    #chat_history.append({"role": "user", "content": prompt})
    #final_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    #print(final_prompt)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    response = requests.post(
        OLLAMA_API_URL, 
        headers=headers, 
        json = payload,
        timeout=120,
        stream=True
    )
    response.raise_for_status()
    full_text = ""

    ## this code gives the response in a streaming manner
    for line in response.iter_lines():
        print(line)
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data = decoded_line[6:]
                if data == "[DONE]":
                    break
                try:
                    json_data = json.loads(data)
                    delta = json_data['choices'][0]['delta']
                    if 'content' in delta:
                        #print(delta['content'], end='', flush=True)
                        content = delta.get("content")

                        if content:
                            full_text += content
                            yield full_text
                except json.JSONDecodeError:
                    continue
    ## end of streaming code

    #data = response.json() #--> this code gives the full response at once
    #return data["choices"][0]["message"]["content"] #--> this code gives the full response at once
    
    '''
    if response.status_code == 200:
       data = response.json()
       print(data)
       
       return data['response']
    
    else:
        return f"Error: {response.status_code} - {response.text}"
    '''
    
## Create Gradio interface

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, label="Enter your code-related question:"),
    outputs=gr.Textbox(lines=10, label="Code Assistant Response:"),
    title="Code Assistant Chatbot",
    description="Ask the Code Assistant any programming-related questions!"
)
if __name__ == "__main__":
    #generate_response("Write a Python function to reverse a string.")
    interface.launch()
  
