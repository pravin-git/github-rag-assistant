
import os

from dotenv import load_dotenv
from fastmcp import Client
from langchain.mcp import MCPAdapter

load_dotenv()

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


async def connect_github():
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise ValueError("GITHUB_TOKEN is missing")

    client = Client(
        GITHUB_MCP_URL,
        auth=token
    )

    adapter = MCPAdapter(client)

    await adapter.__aenter__()

    tools = await adapter.list_tools()

    print("\nAvailable GitHub MCP tools:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    return adapter, tools
