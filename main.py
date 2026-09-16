import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage

from tools import tools, document_search, system_datetime, num_docs


# ============================================================
# SETUP
# ============================================================

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ============================================================
# BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

chat_history = []


# ============================================================
# GET ONLY TEXT
# ============================================================

def get_text(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini content blocks
    if isinstance(content, list):

        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return str(content)


# ============================================================
# ASK AI
# ============================================================

def ask(query):

    # Current question
    messages = chat_history[-10:] + [
        HumanMessage(content=query)
    ]


    # ========================================================
    # FIRST LLM CALL
    # ========================================================

    response = llm_with_tools.invoke(messages)


    # ========================================================
    # TOOL CALL
    # ========================================================

    if response.tool_calls:

        # Add AI tool-call message
        messages.append(response)

        for tool_call in response.tool_calls:

            # -------------------------------
            # Document Search
            # -------------------------------

            if tool_call["name"] == "document_search":

                result = document_search.invoke(
                    tool_call
                )


            # -------------------------------
            # Date / Time
            # -------------------------------

            elif tool_call["name"] == "system_datetime":

                result = system_datetime.invoke(
                    tool_call
                )


            else:

                result = "Unknown tool."

            # Add tool result
            messages.append(result)


        # ====================================================
        # SECOND LLM CALL
        # ====================================================

        response = llm_with_tools.invoke(messages)


    # ========================================================
    # GET ONLY TEXT
    # ========================================================

    answer = get_text(response)


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    chat_history.append(
        HumanMessage(content=query)
    )

    chat_history.append(response)


    return answer


# ============================================================
# START PROGRAM
# ============================================================

print("==================================")
print("SMART AI KNOWLEDGE ASSISTANT")
print("==================================")

print(f"Documents loaded: {num_docs}")

print("Type 'exit' to quit.")


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user = input("\nYou: ")

    if user.lower().strip() in ["exit", "quit"]:

        print("Goodbye!")

        break


    if not user.strip():
        continue


    answer = ask(user)

    print("\nAI:", answer)
