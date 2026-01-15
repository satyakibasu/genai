# RAG Q&A

import os
import streamlit as st
from langchain_groq import ChatGroq # --> will use GROQ Infra for accessing LLM models 
from langchain_openai import OpenAIEmbeddings # --> will use OpenAI embeddings
from langchain_ollama import OllamaEmbeddings # --> will use OpenAI embeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter # --> for splitting into chunks
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #--> creates a chain for passing a list of documents to LLM Model
from langchain_classic.chains import create_retrieval_chain #--> useful if using retrievers
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS #--> vector store from Facebook
from langchain_community.document_loaders import PyPDFDirectoryLoader #--> load pdf documents from a dir
from dotenv import load_dotenv
import time

## Load env variables
load_dotenv()

# Load the GROQ key
groq_api_key = os.getenv('GROQ_API_KEY')

# Create the LLM model
llm = ChatGroq(groq_api_key=groq_api_key,model='llama-3.1-8b-instant')

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context> 
    Question: {input}
    """
)


def create_vector_embeddings():
    """This function will ingest, load into a vector store and piut this into a session variable"""
    
    # Create all these variables and save it into the Streamlit session
    if "vectordb" not in st.session_state: #--> st.session_state is similar to a dict type. So either st.session_state.embeddings or st.session_state['embeddings]
        
        
        st.session_state.embeddings = OpenAIEmbeddings(
                                    model="text-embedding-3-small",
                                    api_key=os.environ["GITHUB_TOKEN"],  # Your GitHub PAT
                                    base_url="https://models.inference.ai.azure.com")
        
        
        #st.session_state['embeddings'] = OllamaEmbeddings(model = 'mistral')

        st.session_state.loader = PyPDFDirectoryLoader('./tenant_agreement') #--> Document ingestion
        st.session_state.docs = st.session_state.loader.load() #--> Document loading into a variable 'docs'.
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectordb = FAISS.from_documents(st.session_state.final_documents,embedding=st.session_state.embeddings)
        

        # No need for return function as this is stored in session variable 

st.title("Rag Chat Application - Tenant Agreement")

user_prompt = st.text_input("Enter your query from the tenant agreements")


if "vectordb" not in st.session_state: #--> very important step
    create_vector_embeddings()
    st.write("Vector database is ready")
    
## This code is the RAG code for retrieving the context of the query.
if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt) #--> pass a list of documents as per the context received
    retriever = st.session_state.vectordb.as_retriever() #--> convert the vectordb as a retriever object
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})

    print(f"Response time: {time.process_time() - start}")
    st.write(response['answer'])

    # With Streamlit Expander creates a container in your app that the user can expand or collapse.
    # we will put the similarity search context under this for more details if expanded
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------")