import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import get_retriever
import tools


# ============================================================
# SETUP
# ============================================================

load_dotenv()
key=os.environ["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=key,
    temperature=0
)


# ============================================================
# RAG
# ============================================================

retriever, num_docs = get_retriever()


# ============================================================
# CONVERSATION MEMORY
# ============================================================

chat_history = []


# ============================================================
# GET TEXT FROM GEMINI
# ============================================================

def get_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content)


# ============================================================
# RUN AGENT
# ============================================================

def ask(query):

    history = "\n".join(
        f"User: {item['user']}\nAI: {item['ai']}"
        for item in chat_history[-10:]
    )

    prompt = f"""
You are a helpful AI assistant.

Conversation history:
{history}

Available tools:

1. document_search
   Use this for questions about the knowledge covered
   by the uploaded PDF.

   IMPORTANT:
   For knowledge questions, ALWAYS search the document
   before answering.

   Do NOT answer from your own knowledge when the document
   may contain the answer.

2. system_datetime
   Use this whenever the user asks for the current
   date, current time, today's date, or current system time.

Important:
- For current time/date questions, ALWAYS use system_datetime.
- For "what was my last question?", use conversation history.
- Do NOT use document_search for conversation questions.
- Answer normally when no tool is needed.

Return ONLY JSON.

If a tool is needed:

{{
    "tool": "tool_name",
    "input": "tool input"
}}

If no tool is needed:

{{
    "tool": null,
    "input": "answer"
}}

User:
{query}
"""

    response = llm.invoke(prompt)
    text = get_text(response)

    # Remove markdown JSON fences
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        decision = json.loads(text)
    except:
        decision = {
            "tool": None,
            "input": text
        }

    tool = decision.get("tool")
    tool_input = decision.get("input", query)


    # ========================================================
    # TOOL: DOCUMENT SEARCH
    # ========================================================

    if tool == "document_search":

        result = tools.document_search.invoke({
            "query": tool_input,
            "retriever": retriever
        })


    # ========================================================
    # TOOL: CURRENT TIME
    # ========================================================

    elif tool == "system_datetime":

        result = tools.system_datetime.invoke({})


    # ========================================================
    # NO TOOL
    # ========================================================

    else:

        result = tool_input


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    if tool is not None:

        final_prompt = f"""
Answer the user's question using the tool result.

User:
{query}

Tool result:
{result}

Give only the final answer.
"""

        final_response = llm.invoke(final_prompt)
        answer = get_text(final_response)

    else:

        answer = result


    # Save conversation
    chat_history.append({
        "user": query,
        "ai": answer
    })

    return answer


# ============================================================
# START
# ============================================================

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