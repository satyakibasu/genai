

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="my-mcp")

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    print("🚀 Starting MCP server...")
    mcp.run(transport="stdio")
