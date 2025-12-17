import asyncio
from typing import List

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def load_github_mcp_tools() -> List[Tool]:
    """
    Connects to the GitHub MCP server via SSE and returns a list of LangChain Tool objects.
    """
    tools: List[Tool] = []

    # Connect to the VS Code MCP GitHub server
    async with sse_client("https://api.githubcopilot.com/mcp/", headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to GitHub MCP server")

            tool_list = await session.list_tools()
            print(f"🛠 Discovered {len(tool_list.tools)} tools")

            # Factory function to correctly capture tool_name for async calls
            def make_tool(tool_name, description=""):
                async def _call_tool(inputs: dict):
                    result = await session.call_tool(tool_name, inputs)
                    # Return the first text output from the tool
                    return result.content[0].text
                return Tool(
                    name=tool_name,
                    description=description,
                    func=_call_tool
                )

            # Build Tool objects for all MCP tools
            for t in tool_list.tools:
                tools.append(make_tool(t.name, t.description or ""))

    return tools


async def main():
    # Load MCP tools
    tools = await load_github_mcp_tools()

    # Create the LLM that will drive the agent
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # Create a LangGraph agent using the MCP tools
    agent = create_react_agent(llm=llm, tools=tools)

    # Example question to the agent
    result = await agent.ainvoke(
        {"messages": [("human", "List the last 5 open PRs in microsoft/autogen")]}
    )

    print("\n🎉 FINAL ANSWER\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())