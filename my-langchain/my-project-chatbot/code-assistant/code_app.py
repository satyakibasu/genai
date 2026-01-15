import requests
import gradio as gr
import json
import time

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codeguru"

def chat_with_ollama(message, history):
    """
    Streaming chat generator for Gradio
    """
    # Build conversation context
    prompt = ""
    for user, assistant in history:
        prompt += f"User: {user}\nAssistant: {assistant}\n"
    prompt += f"User: {message}\nAssistant:"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(
        OLLAMA_API_URL,
        json=payload,
        stream=True,
        timeout=300
    )

    full_response = ""

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        # Handle model warmup
        if data.get("done_reason") == "load":
            time.sleep(1)
            continue

        token = data.get("response", "")
        full_response += token

        yield full_response  # 👈 streaming happens here

def respond(message, history):
    history = history or []
    bot_response = ""

    for partial in chat_with_ollama(message, history):
        bot_response = partial
        yield history + [(message, bot_response)]

with gr.Blocks(title="CodeGuru Assistant") as demo:
    gr.Markdown("## 🧠 CodeGuru – Streaming Code Assistant (Ollama)")

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(
        placeholder="Ask a programming question...",
        show_label=False
    )

    msg.submit(respond, [msg, chatbot], chatbot)
    msg.submit(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch()
