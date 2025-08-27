
import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(override=True)

API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    print("Using GitHub models...")
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
    [("system", "You are a helpful assistant that makes lots of cat references and uses emojis."), ("user", "{input}")]
)
chain = prompt | llm
response = chain.invoke({"input": "write a haiku about a hungry cat that wants tuna 🐱🍣"})
print(f"Response from {API_HOST}: \n")
print(response.content) 