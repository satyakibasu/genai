import asyncio
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


async def main():
    # Start MCP server as a subprocess
    async with stdio_client(
        command="python",
        args=["local_mcp_server.py"],
    ) as (read, write):

        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to MCP server")

            # Discover tools
            tools = await session.list_tools()
            print("🛠 Tools:", [t.name for t in tools.tools])

            # Call tool
            result = await session.call_tool(
                "get_weather",
                {"city": "Pune"}
            )
            print("🌤 Result:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
