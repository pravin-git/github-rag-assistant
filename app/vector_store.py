
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

VECTOR_DB_PATH = "./data/chroma_db"


def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name="github_repository",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    # Simple first-time indexing.
    vector_store.add_documents(chunks)

    print("Repository indexed successfully!")

    return vector_store
