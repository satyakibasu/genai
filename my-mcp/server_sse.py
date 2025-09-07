from fastmcp import FastMCP # for transport="sse"
import requests


mcp = FastMCP(name="my-mcp-sse")

# Create the tools
@mcp.tool()
async def greet(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool()
async def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=3"
    try:
        return requests.get(url, timeout=5).text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Expose over socket
    mcp.run(transport="sse", host="127.0.0.1", port=8765)


