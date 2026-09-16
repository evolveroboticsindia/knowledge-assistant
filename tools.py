from datetime import datetime

from dotenv import load_dotenv
from langchain.tools import tool

from rag import get_retriever


# ============================================================
# SETUP
# ============================================================

load_dotenv()

retriever, num_docs = get_retriever()


# ============================================================
# TOOLS
# ============================================================

@tool
def document_search(query: str) -> str:
    """Search information from the uploaded PDF."""

    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the document."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@tool
def system_datetime() -> str:
    """Get the current date and time."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# TOOL LIST (import this in main.py)
# ============================================================

tools = [
    document_search,
    system_datetime
]
