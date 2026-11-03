
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.nvidia_embeddings import NVIDIAHostedEmbeddings


VECTOR_DB_PATH = "./data/chroma_db"

COLLECTION_NAME = "github_repository_nemotron"


def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No document chunks available for indexing")

    embeddings = NVIDIAHostedEmbeddings(batch_size=1)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    vector_store.add_documents(chunks)

    print("Repository indexed successfully!")

    print(
        "Total vectors:",
        vector_store._collection.count()
    )

    return vector_store
