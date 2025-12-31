from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langserve import add_routes 

load_dotenv()

API_HOST = os.getenv("API_HOST", "github")

if os.getenv("API_HOST", "github") == "github":
    print("Using GitHub models...and model:", os.getenv("GITHUB_MODEL", "openai/gpt-4o"))
    llm = ChatOpenAI(
        model_name=os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
        openai_api_base="https://models.github.ai/inference",
        openai_api_key=os.environ["GITHUB_TOKEN"],
    )
elif API_HOST == "ollama":
    print("Using Ollama model on local...")
    llm = ChatOpenAI(
        model_name=os.getenv("OLLAMA_MODEL", "mistral"),
        openai_api_base=os.environ["OLLAMA_ENDPOINT"],
        openai_api_key="nokeyneeded",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Language Translator. Translate into the following {language}."), 
        ("user", "{input}")
    ]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Application Definition
app = FastAPI(title="This is my Langchain server",
              version="1.0",
              description="Simple API Server for Langchain interfaces")

# Adding chain routes
add_routes(app,chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='localhost',port=8000)