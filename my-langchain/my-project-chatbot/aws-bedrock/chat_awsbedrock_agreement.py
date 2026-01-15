## This code will use AWS Bedrock to create a chatbot application using Streamlit and LangChain
## Langchain provides integration with AWS Bedrock for building LLM applications.

import streamlit as st
import boto3
from typing import Union
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Initialize Bedrock Embeddings
embeddings = BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v2:0")


# Function to ingest documents and create vector store
def ingest_documents() -> FAISS:
    """This function will ingest documents from a directory and create a FAISS vector store"""

    # Load PDF documents from the specified directory
    loader = PyPDFDirectoryLoader("tenant_agreement")  # specify your directory here
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Number of document chunks: {len(docs)}")

    # Create a FAISS vector store from the documents
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")  # Save the vector store locally

    return vector_store


# Function to create an LLM model from different providers using Bedrock
def create_llm_model(model_name: str) -> Union[BedrockLLM, ChatBedrock]:
    """Create appropriate LLM model based on model name"""
    
    # Chat-based models (Anthropic, Mistral, DeepSeek)
    if model_name.startswith(("anthropic.", "mistral.", "deepseek.","a21.jamba-")):
        return ChatBedrock(
            model_id=model_name,
            client=client,
            region_name="us-east-1",
            model_kwargs={
                "temperature": 0.2,
                "max_tokens": 512  # Changed from maxTokenCount
            }
        )

    # AI21 models
    elif model_name.startswith("ai21.j2"):
        return BedrockLLM(
            model_id=model_name,
            client=client,
            region_name="us-east-1",
            model_kwargs={
                "temperature": 0.2,
                "maxTokens": 512
            }
        )

    # Amazon Titan Text models
    elif model_name.startswith("amazon.titan-text"):
        return BedrockLLM(
            model_id=model_name,
            client=client,
            region_name="us-east-1",
            model_kwargs={
                "temperature": 0.2,
                "maxTokenCount": 512
            }
        )

    else:
        raise ValueError(f"Unsupported or non-text model: {model_name}")


def create_prompt(model_name: str):
    """Create appropriate prompt template based on model type"""
    # Most modern models are chat-based, so we check for text completion models instead
    if model_name.startswith(("amazon.titan-text", "ai21.j2", "cohere.command")):
        # Text completion models
        return PromptTemplate.from_template("""
            You are a helpful assistant.
            Use the context below to answer the question.
            If you don't know, say you don't know.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """)
    else:
        # Chat-based models (Anthropic, Mistral, Meta, DeepSeek, AI21 Jamba, etc.)
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context to answer the question. If you don't know the answer, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# Function to generate response from the LLM model
def generate_response(question: str, model_name: str, vector_store: FAISS) -> str:
    """This function will generate and send the response back using Bedrock LLM"""
    
    llm_model = create_llm_model(model_name)
    print(f"Using model: {llm_model}")

    prompt = create_prompt(model_name)
    retriever = vector_store.as_retriever()
    
    # Test retrieval
    results = vector_store.similarity_search(question)
    print(f"Top retrieved chunk: {results[0].page_content[:200]}...")

    # Create a retrieval chain - SIMPLIFIED VERSION
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x # Pass through the question. This evaluates at run time
        }
        | prompt
        | llm_model
        | StrOutputParser()
    )

    # Get the answer from the chain
    answer = rag_chain.invoke(question)

    return answer


## Streamlit App

st.title("AWS Bedrock RAG Chatbot")


model_name = st.selectbox(
    "Select Bedrock Model:",
    options=[
        "mistral.mistral-7b-instruct-v0:2",
        "anthropic.claude-2:2024-06-19",
        "amazon.titan-text-v2:0",
        "ai21.j2-jumbo-instruct:3"
    ]
)

# Let's create a sidebar to update or create the vector store.
st.sidebar.title("Vector Store Management")



# Load the vector store from local if not already created in this session

if st.sidebar.button("Ingest Tenant Agreement Documents"):
    with st.sidebar.spinner("Ingesting documents and creating vector store..."):
        vector_store = ingest_documents()
    st.sidebarsuccess("Vector store created/updated successfully!")

vector_store = FAISS.load_local(
                "faiss_index", 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
        )


user_input = st.text_input("Enter your question about the tenant agreement:")

if user_input:
    response = generate_response(user_input, model_name, vector_store)
    st.write("Model Name:" ,model_name)
    st.write(response)



if __name__ == "__main__":
    try:
        # Option 1: Ingest documents and create vector store (first time)
        # vector_store = ingest_documents()
        
        # Option 2: Load the vector store from local (subsequent runs)
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        ) 

        # Test the RAG system
        question = "Who are the parties involved?"
        model_name = "mistral.mistral-7b-instruct-v0:2"  # Updated to a valid model ID
        
        print(f"\nQuestion: {question}")
        answer = generate_response(question, model_name, vector_store)
        print(f"\nAnswer: {answer}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()