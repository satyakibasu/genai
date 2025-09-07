import asyncio
from mcp.client.stdio import stdio_client,StdioServerParameters
import os

from mcp.client.session import ClientSession


async def main():
    script_path = os.path.abspath("server_stdio.py")
    server_params = StdioServerParameters(command="python",args=[script_path],env=None)
    
  # Everything happens inside these async with blocks
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Session initialized")

            # List tools
            response = await session.list_tools()
            print("Available tools:", response.tools)

            for t in response.tools:
                print(f"Tool Name: {t.name}, Description: {t.description}")
            

            # Call greet
            response = await session.call_tool("greet", {"name": "Alice"})
            print("Response:", response.content[0].text)


asyncio.run(main())