
import asyncio

from app.github_client import connect_github
from app.ingestion import fetch_repository_files
from app.vector_store import create_vector_store
from app.rag import create_rag_chain


async def main():

    # 1. Connect to GitHub MCP
    adapter, tools = await connect_github()

    try:

        # 2. Fetch source code
        documents = await fetch_repository_files(tools)

        if not documents:
            raise RuntimeError(
                "No repository files were fetched"
            )

        # 3. Create vector database
        vector_store = create_vector_store(documents)

        # 4. Create RAG engine
        ask = create_rag_chain(vector_store)

        # 5. Start interactive chat
        print("\nGitHub RAG Assistant Ready!")
        print("Type 'exit' to quit.")

        while True:

            question = input("\nAsk: ")

            if question.lower() == "exit":
                break

            result = ask(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nSource files:")
            for source in result["sources"]:
                print(f"- {source}")

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
