import os
import streamlit as st
from dotenv import load_dotenv
from rich import print

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ----------------- Load environment -----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ----------------- LLM -----------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    streaming=True
)

# ----------------- Tools -----------------
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

duck_search = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the internet using DuckDuckGo"
)

tools = [wiki_tool, arxiv_tool, duck_search]

# ----------------- Streamlit UI -----------------
st.title("Web Search Chatbot with Tools")
st.sidebar.title("Settings")

# Button to clear chat history
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = []

# Initialize session_state messages if not present
if "messages" not in st.session_state or not st.session_state["messages"]:
    st.session_state["messages"] = [
        AIMessage(content="Hi, I am a chatbot who can search the web. How can I help you?")
    ]

# Render chat history
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ----------------- Chat Input -----------------
user_input = st.chat_input("Type your query here...")
if user_input:
    # Add user message
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    st.chat_message("user").write(user_input)

    # ----------------- Agent -----------------
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can use the provided tools to answer user queries."
    )

    # StreamlitCallbackHandler to show intermediate agent thoughts
    callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

    # Convert session messages to dicts for agent input. The session messages contain HumanMessage and AIMessage objects.
    chat_input = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
    for m in st.session_state["messages"]
    ]

    # Invoke agent with messages
    ai_response = agent.invoke(
    {"messages": chat_input},  # Must be a dict
    config={"callbacks": [callback]}
    )

    # Extract AI message content
    ai_msg_content = ai_response['messages'][-1].content

    # Append AI response to session state
    st.session_state["messages"].append(AIMessage(content=ai_msg_content))
    
    # Display AI response
    st.chat_message("assistant").write(ai_msg_content)
