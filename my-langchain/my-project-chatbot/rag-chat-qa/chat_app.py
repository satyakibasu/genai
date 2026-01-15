import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser # --> To parse the LLM response
from langchain_core.prompts import ChatPromptTemplate # --> for chat template
import os
from dotenv import load_dotenv

# Load and set env variables
load_dotenv()
os.environ["API_HOST"] = os.getenv("API_HOST", "github")

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "chat-with-openai-ollama" #os.getenv("LANGCHAIN_PROJECT", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to user queries"),
        ("user","Question:{question}") 
    ]
)

def generate_response(question,provider,model_name,temperarture,max_tokens)->str:
    """This function will generate and send the response back"""
    
    if provider == 'OpenAI':

        model = "openai/"+ model_name
        llm = ChatOpenAI(
            model_name = model, #os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
            openai_api_base="https://models.github.ai/inference",
            openai_api_key=os.environ["GITHUB_TOKEN"])
    elif provider == 'Ollama':
        llm = ChatOllama(model=model_name)
        
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})

    return answer


## Title of the App
st.title("Q&A Chatbot")

# Set the sidebar parameters
st.sidebar.title("Settings")
provider = st.sidebar.radio("Choose one option:",["OpenAI", "Ollama","Gemini"])

if provider == "OpenAI":
    models = ["gpt-4o", "gpt-4o-mini"]
elif provider == "Ollama":
    models = ["mistral", "llama3", "phi"]
else:
    models = ["gemini-1.5-pro", "gemini-1.5-flash"]

model_name = st.selectbox("Select model",models)
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(provider=provider,question=user_input, model_name=model_name,temperarture=temperature,max_tokens=max_tokens)
    st.write("Model Name:" ,model_name)
    st.write(response)
else:
    st.write("please provide the query")
    
