import streamlit as st
from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.tools import WikipediaQueryRun
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq #--> using the GROQ Infra where different models are hosted.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

@tool
def calculator_tool(vars:list) -> float:
    """A tool for performing mathematical calculations. Only a list of numbers is provided as input."""

    return sum(vars)  # Sum up the list of numbers provided as input.
    

#Load environment variables
load_dotenv()

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
    llm = ChatGroq(groq_api_key=groq_api_key,model='Gemma-9b-It')


# Initialize the agent with Wikipedia tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Tool for searching internet and solving Math problems.")
tools = [wiki_tool, calculator_tool]

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful math assistant. Use the tools provided to answer user queries accurately."),
    HumanMessage(content="{input}")
])

# Create the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful math assistant. Use the tools provided to answer user queries accurately."
)

## ------- Streamlit App Interface ------- ##
st.title("Maths GPT")
st.subheader("Ask any Maths related question")
st.sidebar.title("Message History")

# Initialize session state for messages if not present
if "messages" not in st.session_state or not st.session_state["messages"]:
    st.session_state["messages"] = [AIMessage(content="Hello! I am Maths GPT. Ask me any Maths related question.")]

st.sidebar.write(st.session_state["messages"]) #--> for debugging purpose

user_input = st.chat_input("Type your query here...")
#Display the history of messages on the main screen
for msg in st.session_state["messages"]:  # Exclude the latest user message
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

if user_input:
    # Add user message
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    st.chat_message("user").write(user_input)

    # Convert session messages to dicts for agent input. The session messages contain HumanMessage and AIMessage objects.
    chat_input = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in st.session_state["messages"]
        ]



    # Invoke agent with messages
    ai_response = agent.invoke(
        {"messages": chat_input},  # Must be a dict
        )
    # Extract AI message content
    ai_msg_content = ai_response['messages'][-1].content

    # Append AI response to session state
    st.session_state["messages"].append(AIMessage(content=ai_msg_content))
    st.sidebar.write(st.session_state["messages"]) #--> for debugging purpose
        
    # Display AI response
    st.chat_message("assistant").write(ai_msg_content)

    

   