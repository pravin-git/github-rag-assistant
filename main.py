
import asyncio
import traceback

from app.github_client import connect_github
from app.ingestion import fetch_repository_files
from app.vector_store import create_vector_store
from app.rag import create_rag_chain


async def main():

    print("=" * 60)
    print("       GITHUB REPOSITORY RAG ASSISTANT")
    print("=" * 60)

    try:

        # Step 1: Connect to GitHub MCP
        print("\n[1/5] Connecting to GitHub MCP...")

        async with connect_github() as (adapter, tools):

            print("GitHub MCP connected successfully!")

            # Step 2: Fetch repository files
            print("\n[2/5] Fetching repository files...")

            documents = await fetch_repository_files(tools)

            if not documents:
                raise RuntimeError(
                    "No repository documents found. "
                    "Check GitHub repository configuration."
                )

            print(
                f"Successfully fetched "
                f"{len(documents)} documents."
            )

            # Step 3: Generate embeddings and store
            print("\n[3/5] Creating NVIDIA embeddings...")

            vector_store = create_vector_store(documents)

            print("ChromaDB indexing completed!")

            # Step 4: Initialize RAG engine
            print("\n[4/5] Initializing RAG engine...")

            ask = create_rag_chain(vector_store)

            print("RAG engine initialized successfully!")

            # Step 5: Interactive chat
            print("\n[5/5] Starting GitHub RAG Assistant...")

            print("\n" + "=" * 60)
            print("Assistant is ready!")
            print("Type 'exit' or 'quit' to stop.")
            print("=" * 60)

            while True:

                try:

                    question = input("\nYou: ").strip()

                    if not question:
                        continue

                    if question.lower() in ("exit", "quit"):
                        print("\nThank you for using GitHub RAG Assistant!")
                        break

                    print("\nSearching repository...")

                    result = ask(question)

                    print("\n" + "-" * 60)
                    print("AI RESPONSE")
                    print("-" * 60)

                    print(result["answer"])

                    print("\n" + "-" * 60)
                    print("SOURCE FILES")
                    print("-" * 60)

                    sources = result.get("sources", [])

                    if sources:
                        for source in sources:
                            print(f"- {source}")
                    else:
                        print("No source files identified.")

                    print("-" * 60)

                except KeyboardInterrupt:
                    print("\nChat interrupted.")
                    break

                except Exception as error:
                    print(f"\nError processing question: {error}")
                    traceback.print_exc()

    except KeyboardInterrupt:
        print("\nApplication interrupted.")

    except Exception as error:
        print("\nAPPLICATION ERROR")
        print("-" * 60)
        print(f"Error: {error}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
