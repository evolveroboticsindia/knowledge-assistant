import datetime
from langchain.tools import tool


@tool
def document_search(query: str, retriever) -> str:
    """Search information from the uploaded PDF."""

    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the document."

    return "\n\n".join(
        doc.page_content[:800]
        for doc in docs[:3]
    )


@tool
def system_datetime() -> str:
    """Return the current system date and time."""

    return datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

