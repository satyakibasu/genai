# This code will use both ai agents and MCP server in conjunction
# The MCP server used is server_sse.py.

import asyncio
import base64
import os
from dotenv import load_dotenv
import openai
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from rich import print
import json

load_dotenv()   

API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
if API_HOST == "github":
    asynch_client = openai.AsyncOpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])   



# Let's write the client code to connect to the MCP server using SSE transport.
# This connects to an MCP server running at http://http://127.0.0.1:8765/sse". The name of the MCP server is "my-mcp-sse".
async def create_chat_completion(user_message: str):
    async with sse_client("http://127.0.0.1:8765/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized")

            # ----------------------------------------------------------------
            # Step 1: Lets try to get the info about the MCP server tools
            # ----------------------------------------------------------------

            # List tools of the MCP server
            tools = await session.list_tools()
            #print("Tools received from MCP server:", tools)

            # Convert MCP tools to OpenAI function calling format
            openai_tools = convert_mcp_tools_to_openai(tools)
            #print("Converted tools for OpenAI:", openai_tools)
            

            # ----------------------------------------------------------------
            # Step 2: Pass this to the LLM and get the function call using tools.
            # ----------------------------------------------------------------

            # Now create an LLM call and pass the functions in tools
            response = await asynch_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. You can use the {tools} to answer user queries."},
                    {"role": "user", "content": user_message}
                ],
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.7,
            )
            print(f"Response from {MODEL_NAME} on {API_HOST}")
            print("AI Response:", response.choices[0].message)


            # ----------------------------------------------------------------
            # Step 3: Check for function call and invoke the tool using MCP session to the MCP server.
            # ----------------------------------------------------------------

            # Check if a tool was called
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0].function
                tool_name = tool_call.name
                tool_args = tool_call.arguments 
                
                # Convert from JSON string -> Python dict
                tool_args = json.loads(tool_args) if tool_args else {}

                print(f"GPT will call function: {tool_name} with arguments: {tool_args}")

                # Now we have to call the tool using MCP session (session is important)
                result = await session.call_tool(tool_name, tool_args)
                print("Result from weather tool:", result.content[0].text)


            # ----------------------------------------------------------------
            # Step 4: Append the user message and tool response to messages and call LLM again
            # ----------------------------------------------------------------

                # Now let's append the user message and tool response to messages and call LLM again
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. You can use the {tools} to answer user queries."},
                    {"role": "user", "content": user_message}]
                
                messages.append(response.choices[0].message) # append the message received from LLM in the previous call
                messages.append({
                    "role":"tool",
                    "tool_call_id":response.choices[0].message.tool_calls[0].id,
                    "content":result.content[0].text,
                })

                follow_up_response = await asynch_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    tools=openai_tools,
                )
                print("Follow-up AI Response:", follow_up_response.choices[0].message.content)

        

# This function converts MCP tools to OpenAI function calling format.
def convert_mcp_tools_to_openai(mcp_tools):
    openai_tools = []
    for tool in mcp_tools.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
            }
        })
    return openai_tools


if __name__ == "__main__":
    #user_message = "What's the weather like in New York City today?"
    user_message = "How do i greet Alice?"
    
    asyncio.run(create_chat_completion(user_message))