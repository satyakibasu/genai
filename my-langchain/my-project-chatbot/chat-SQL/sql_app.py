import streamlit as st
from pathlib import Path
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import os
import sqlite3
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  


# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM with Groq API key
"""
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    streaming=True
)
"""
llm = ChatOpenAI(
    model=os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
    openai_api_base="https://models.github.ai/inference",
    openai_api_key=os.environ["GITHUB_TOKEN"]
)


# Define the path to the SQLite database
LOCAL_DB_PATH = 'student.db'

# Create a connection to the SQLite database
db = SQLDatabase.from_uri(f"sqlite:///{LOCAL_DB_PATH}")

# Create the SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm) #--> this is required for agent to interact with SQL db
tools = toolkit.get_tools() #--> this will give us the list of tools to interact with SQL db

# Create the SQL agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are an agent designed to interact with a SQL database. "
        "Use the tools to query the database and answer user questions."
        "Explain your answer step-by-step, but do not reveal internal reasoning"
    ),
)

# Streamlit app title
st.title("SQL Chatbot with Groq and LangChain")
st.set_page_config(page_title="SQL Chatbot using SQLite DB", page_icon=":robot_face:")
st.sidebar.title("Message History (for debugging)")
st.write("This application allows you to chat with a SQL database using natural language.The database contains student information. Ask any questions related to the student database!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({
        "role": "assistant",
        "content": "Hello! I am your SQL chatbot. How can I assist you with the student database today?"
    })



# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])
        #st.write(message["content"])

# Chat input box
if user_input := st.chat_input("Type your query about the student database..."):
    # Display user message in chat
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    st.sidebar.write(st.session_state["messages"])

    
    # Get the agent's response
    with st.chat_message("assistant"):
        # Create a callback handler for Streamlit
        #streamlit_callback = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True) #--> LLM providers does not support

        response = agent.invoke(
            {"messages": st.session_state["messages"]},
            #callbacks=[streamlit_callback]
        )
        #st.markdown(response)

        # Extract AI message content
        ai_msg_content = response['messages'][-1].content

        
        # Display AI response
        #st.chat_message("assistant").write(ai_msg_content)
        st.write(ai_msg_content)

        # Append AI response to session state
        st.session_state["messages"].append({"role": "assistant", "content": ai_msg_content})