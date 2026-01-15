import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq #--> using the GROQ Infra where different models are hosted.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import validators
from langchain_core.prompts import PromptTemplate #--> used for text or string inputs
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# Load environment variables from .env file
load_dotenv()
os.environ["YOUTUBE_API_KEY"] = os.getenv("YOUTUBE_API_KEY")

# Initialize the ChatOpenAI model
if os.getenv("API_HOST") == "github":
    llm = ChatOpenAI(
                model_name = os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
                openai_api_base="https://models.github.ai/inference",
                openai_api_key=os.environ["GITHUB_TOKEN"])
elif os.getenv("API_HOST") == "groq":
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=groq_api_key,model='llama-3.1-8b-instant')
elif os.getenv("API_HOST") == "huggingface":
    
    repo_id = "deepseek-ai/DeepSeek-R1-0528"

    llm_base = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        provider="auto",  # set your provider here hf.co/settings/inference-providers
    )
    llm = ChatHuggingFace(llm=llm_base)


## ------- Streamlit App Interface ------- ##

st.title("Text Summarization App")
st.subheader("Summarize text from URLs or direct input")
st.write(f"Using Inference Provider: {os.getenv('API_HOST')}")

url_input = st.text_input("Enter the URL to summarize:", label_visibility="collapsed")

if st.button("Summarize URL"):
    if validators.url(url_input):
        # Loading the website data
        if "youtube.com" in url_input or "youtu.be" in url_input:
            loader = YoutubeLoader.from_youtube_url(url_input, add_video_info=False)
        else:
            loader = UnstructuredURLLoader(urls=[url_input],ssl_verify=False, 
                                           headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15"})  

        
        documents = loader.load()
        text_content = " ".join([doc.page_content for doc in documents])

        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text in 300 words:\n\n{text}\n\nSummary:"
        )

        chain = prompt | llm
        formatted_prompt = prompt.format(text=text_content)
        response = chain.invoke(formatted_prompt)

        #response = llm.invoke([HumanMessage(content=formatted_prompt)])
        summary = response.content
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter a valid URL.")
