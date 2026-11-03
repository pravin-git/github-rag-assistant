
import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def create_rag_chain(vector_store):

    llm = ChatNVIDIA(
        model=os.getenv(
            "NVIDIA_CHAT_MODEL",
            "nvidia/nemotron-3-super-120b-a12b"
        ),
        temperature=0
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a senior software architect.

            Answer questions using ONLY the
            provided repository context.

            Instructions:
            - Do not invent code or functionality.
            - Explain the implementation clearly.
            - Mention relevant file paths.
            - If context is insufficient,
              explicitly say so.
            - Identify dependencies between files
              when supported by the context.

            Repository context:

            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    def ask(question):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [
                f"File: {doc.metadata.get('source')}\n"
                f"Code:\n{doc.page_content}"
                for doc in docs
            ]
        )

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        sources = list(set(
            doc.metadata.get("source")
            for doc in docs
        ))

        return {
            "answer": response.content,
            "sources": sources
        }

    return ask
