# RAG Q&A with chat history and upload feature

import os
import streamlit as st
from langchain_classic.chains import create_history_aware_retriever #--> retriever knows the chat history
from langchain_classic.chains import create_retrieval_chain #--> useful if using retrievers
from langchain_text_splitters import RecursiveCharacterTextSplitter # --> for splitting into chunks
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #--> creates a chain for passing a list of documents to LLM Model
from langchain_chroma import Chroma #--> Vector store
from langchain_community.chat_message_histories import ChatMessageHistory #--> used to store the chat history
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate #--> chat prompt template
from langchain_core.prompts import MessagesPlaceholder #--> variable to hold the history of converstaions outside of prompt. This is injected to the prompt.
from langchain_groq import ChatGroq #--> using the GROQ Infra where different models are hosted.
from langchain_huggingface import HuggingFaceEmbeddings #--> using Huggingface embeddings
from langchain_community.document_loaders import PyPDFLoader #--> load pdf documents from a dir
from langchain_text_splitters import RecursiveCharacterTextSplitter # --> for splitting into chunks
from langchain_core.runnables import RunnableWithMessageHistory # --> automatically appends the history to the MessagePlaceHolder without any manual input.
from rich import print

from dotenv import load_dotenv

## Load env variables
load_dotenv()

# Create the LLM model
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key,model='llama-3.1-8b-instant')

# Create the embeddings using HuggingFace
HF_TOKEN = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Setup the streamlit app
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDF and chat with their content")

session_id = st.text_input("Session Id", value="default_session")

# Statefully Manage Chat History
if 'store' not in st.session_state:
    st.session_state.store = {} 

uploaded_files = st.file_uploader("Choose a PDF file to upload", type="pdf",accept_multiple_files=True)

# Process the uploaded files
if uploaded_files:
    documents = [] #--> This will hold all the 'Document' which will go into the Splitter and Vector for embeddings 
    for uploaded_file in uploaded_files:
        with open('temp.pdf','wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader('temp.pdf')
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_documents = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=final_documents, embedding=embeddings)
    retriever = vectordb.as_retriever()

    context_system_prompt = """
    Given a chat history and latest user question, which might reference the context history
    formulate a standalone question which can be understood.
    Without chat history, do NOT answer the question.
    Just reformulate it if needed and other return as it is  
    """

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,prompt)

    # Answer question prompt
    system_prompt = """
    You are an assistant for question answer tasks.
    Use the following pieces of context to answer the questions.
    If you do not know the answer, say you do not know. 
    Use three sentences to keep the answer concise.
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()

        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,get_session_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    # User Input
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input":user_input},
            config={"configurable":{"session_id":session_id}},
        )
    
        st.write("Assistant:",response['answer'])
        #st.write(st.session_state.store)
        st.write("Chat History:",session_history.messages)

        print(st.session_state.store)
    
    



