import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage

from rag import get_retriever
from tools import create_tools


# =========================
# SETUP
# =========================

load_dotenv()

retriever, num_docs = get_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# =========================
# TOOLS
# =========================

tools = create_tools(retriever)

llm_with_tools = llm.bind_tools(tools)


# =========================
# HISTORY
# =========================

chat_history = []


# =========================
# GET TEXT
# =========================

def get_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return str(content)


# =========================
# ASK AI
# =========================

def ask(query):

    messages = chat_history[-10:] + [
        HumanMessage(content=query)
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            # Find the correct tool
            tool = next(
                t for t in tools
                if t.name == tool_call["name"]
            )

            result = tool.invoke(tool_call)

            messages.append(result)

        response = llm_with_tools.invoke(messages)

    answer = get_text(response)

    chat_history.append(
        HumanMessage(content=query)
    )

    chat_history.append(response)

    return answer


# =========================
# START
# =========================

print("==================================")
print("SMART AI KNOWLEDGE ASSISTANT")
print("==================================")

print(f"Documents loaded: {num_docs}")
print("Type 'exit' to quit.")


while True:

    user = input("\nYou: ")

    if user.lower().strip() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user.strip():
        continue

    answer = ask(user)

    print("\nAI:", answer)
