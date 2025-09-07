# this uses mcp = 1.13 sdk
# ensure that server_sse.py is running before executing this client code

import asyncio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client   



async def main_sse(client_name: str):
    async with sse_client("http://127.0.0.1:8765/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Session initialized")
            print(f"{client_name} Connected!")

            # List tools
            response = await session.list_tools()
            print("🛠 Available tools:", response.tools)

            # Call weather tool
            result = await session.call_tool("get_weather", {"city": "Pune"})
            print("🎉 Result from weather tool:", result.content[0].text)


async def main():
    # simulate 3 clients connecting in parallel
    await asyncio.gather(
        main_sse("Client-1"),
        main_sse("Client-2"),
        main_sse("Client-3"),
    )

if __name__ == "__main__":
    asyncio.run(main())