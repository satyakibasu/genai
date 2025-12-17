import asyncio
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("local-mcp")

# Register a tool
@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"The weather in {city} is sunny, 25°C"


if __name__ == "__main__":
    # Runs on stdio by default (correct MCP behavior)
    asyncio.run(mcp.run())
