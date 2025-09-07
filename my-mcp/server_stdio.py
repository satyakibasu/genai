#from fastmcp import FastMCP # for transport="sse"
from mcp.server.fastmcp import FastMCP # for transport="stdio"
import requests

mcp = FastMCP(name="my-mcp")

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool()
def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=3"
    try:
        return requests.get(url, timeout=5).text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    #print("🚀 Starting MCP server on http://127.0.0.1:8765/sse ...")
    #mcp.run(transport="sse", host="127.0.0.1", port=8765)
    #mcp.run(transport="sse")

    print("🚀 Starting MCP stdio server...")
    mcp.run(transport="stdio")