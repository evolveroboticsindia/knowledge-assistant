# 🧠 Smart AI Knowledge Assistant

A simple AI assistant built with **LangChain, Google Gemini, FAISS, and RAG**.

It can:

* 📄 Answer questions from uploaded PDFs
* 🔍 Retrieve relevant document information
* 🕐 Get the current date and time
* 💬 Maintain recent conversation history
* 🧰 Use custom tools automatically

## 🏗️ Architecture

```text
User
 ↓
Gemini LLM
 ↓
Tool Decision
 ├── document_search → Retriever → FAISS → PDF
 └── system_datetime → Current Date/Time
 ↓
Gemini LLM
 ↓
Final Answer
```

## 📁 Project Structure

```text
project/
├── main.py
├── rag.py
├── tools.py
├── requirements.txt
├── .env
└── documents/
    └── *.pdf
```

## 📦 Installation

```bash
pip install -U \
langchain \
langchain-community \
langchain-text-splitters \
langchain-google-genai \
faiss-cpu \
pypdf \
python-dotenv
```

## 🔑 API Key

Create `.env`:

```env
GOOGLE_API_KEY=your_api_key
```

## ▶️ Run

Place PDFs inside `documents/` and run:

```bash
python main.py
```

Example:

```text
You: What is machine learning?

AI: [Answer based on the uploaded PDF]
```

## 🔄 RAG Pipeline

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Chunks
 ↓
Gemini Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Relevant Chunks
 ↓
Gemini
 ↓
Answer
```

### Key Components

* **Gemini** → LLM and embeddings
* **FAISS** → Stores/searches vectors
* **Retriever** → Retrieves relevant document chunks
* **RAG** → Provides document context to the LLM
* **Tools** → `document_search` and `system_datetime`

## 📌 Files

| File         | Purpose                                     |
| ------------ | ------------------------------------------- |
| `main.py`    | Main AI assistant and conversation handling |
| `rag.py`     | PDF loading, chunking, embeddings and FAISS |
| `tools.py`   | Custom tools                                |
| `documents/` | PDF knowledge base                          |
| `.env`       | API key                                     |

## 🚀 Future Improvements

* Save/load FAISS index
* Web interface
* Persistent memory
* Source/page citations
* More tools
