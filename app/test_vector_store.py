
import asyncio

from app.github_client import connect_github
from app.ingestion import fetch_repository_files
from app.vector_store import create_vector_store


async def main():

    # Step 1: Connect to GitHub MCP
    adapter, tools = await connect_github()

    try:

        # Step 2: Fetch repository documents
        print("\nFetching repository files...")

        documents = await fetch_repository_files(tools)

        print(f"Documents fetched: {len(documents)}")

        if not documents:
            raise RuntimeError("No documents fetched")

        # Step 3: Create ChromaDB vector store
        print("\nCreating ChromaDB vector store...")

        vector_store = create_vector_store(documents)

        # Step 4: Check collection
        collection = vector_store._collection

        print("\nChromaDB Statistics")
        print("-------------------")
        print("Collection:", collection.name)
        print("Total chunks:", collection.count())

        # Step 5: Test semantic search
        query = "Explain the application architecture"

        print("\nSearching for:", query)

        results = vector_store.similarity_search(
            query,
            k=3
        )

        print("\nSearch Results")
        print("--------------")

        for index, doc in enumerate(results, start=1):

            print(f"\nResult {index}")
            print("Source:", doc.metadata.get("source"))
            print("Content:")
            print(doc.page_content[:500])

        print("\nChromaDB test completed successfully!")

    finally:
        # MCPAdapter has no close() method in your
        # currently installed version.
        pass


if __name__ == "__main__":
    asyncio.run(main())
