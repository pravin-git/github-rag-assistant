
import os

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


def extract_text(result):
    """
    Extract text from MCP tool output.
    Handles common text-block and dictionary responses.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, list):
        return "\n".join(
            extract_text(item) for item in result
        )

    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])

        if "content" in result:
            return extract_text(result["content"])

        return str(result)

    return str(result)


async def fetch_repository_files(tools):

    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", "main")

    # Start with a few known source files.
    file_paths = [
        "README.md",
        "package.json",
        "src/App.jsx",
    ]

    github_tool = next(
        (
            tool for tool in tools
            if tool.name.endswith("get_file_contents")
        ),
        None
    )

    if github_tool is None:
        raise RuntimeError(
            "get_file_contents MCP tool not found"
        )

    documents = []

    for path in file_paths:

        try:
            result = await github_tool.ainvoke({
                "owner": owner,
                "repo": repo,
                "path": path,
                "ref": f"refs/heads/{branch}"
            })

            content = extract_text(result)

            if not content.strip():
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path,
                        "repository": f"{owner}/{repo}",
                        "branch": branch
                    }
                )
            )

            print(f"Fetched: {path}")

        except Exception as error:
            print(f"Skipping {path}: {error}")

    return documents
