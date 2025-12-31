from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Simple GenAI Application

# Load the environment variables
load_dotenv(override=True)

os.environ["API_HOST"] = os.getenv("API_HOST", "github")

# Langchain Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

API_HOST = os.getenv("API_HOST", "github")
#API_HOST = "ollama"

# Define the LLM model
if API_HOST == "github":
    print("Using GitHub models...and model:", os.getenv("GITHUB_MODEL", "openai/gpt-4o"))
    llm = ChatOpenAI(
        model_name=os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
        openai_api_base="https://models.github.ai/inference",
        openai_api_key=os.environ["GITHUB_TOKEN"],
    )
elif API_HOST == "ollama":
    print("Using Ollama model on local...")
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assitant. Provide me answers based on the questions.."), 
        ("user", "Question: {question}")
    ]
)

## Streamlit Framework
st.title("Chat Bot Application")
input_text = st.text_input("What is the question you have?")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))