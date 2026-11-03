
from app.nvidia_embeddings import NVIDIAHostedEmbeddings


def main():

    embeddings = NVIDIAHostedEmbeddings()

    documents = [
        "React is a JavaScript library for building user interfaces.",
        "Redux is used for application state management.",
        "Next.js supports server-side rendering."
    ]

    print("Testing document embeddings...")

    vectors = embeddings.embed_documents(documents)

    print("Documents:", len(vectors))
    print("Dimensions:", len(vectors[0]))

    print("\nTesting query embedding...")

    query_vector = embeddings.embed_query(
        "Explain React application architecture"
    )

    print("Query dimensions:", len(query_vector))

    print("\nEmbedding integration successful!")


if __name__ == "__main__":
    main()
