
import asyncio

from app.github_client import connect_github
from app.ingestion import fetch_repository_files


async def main():

    adapter, tools = await connect_github()

    try:
        print("\nFetching repository files...\n")

        documents = await fetch_repository_files(tools)

        print(f"\nTotal documents fetched: {len(documents)}")

        for doc in documents:

            print("\n" + "=" * 60)
            print(f"File: {doc.metadata['source']}")
            print(f"Repository: {doc.metadata['repository']}")
            print("=" * 60)

            print(doc.page_content[:500])

    finally:
        # MCPAdapter does not expose close().
        # Its underlying MCP client manages the connection.
        pass


if __name__ == "__main__":
    asyncio.run(main())
