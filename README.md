# 🤖 LangChain Intelligent Chatbot

A production-ready, modular chatbot built with **LangChain**, **OpenAI**, and **Streamlit**.  
Features general conversation with memory + PDF document question-answering (RAG).

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 General Chat | Multi-turn conversation with context memory |
| 📄 PDF Q&A (RAG) | Upload PDFs and ask questions about their content |
| 🧠 Conversation Memory | Remembers previous messages using `RunnableWithMessageHistory` |
| 🎨 Polished UI | Dark-themed Streamlit interface with streaming responses |
| 🔒 Secure | API keys loaded from `.env`, never hardcoded |
| 📦 Modular | Clean separation of concerns across 7 modules |

---

## 🏗️ Architecture

```
app.py (Streamlit UI)
├── chatbot.py (General Chat Chain)
│   ├── prompts.py (Prompt Templates)
│   ├── llm_setup.py (LLM Factory)
│   └── memory.py (Session History)
├── rag.py (PDF RAG Chain)
│   ├── prompts.py
│   ├── llm_setup.py
│   └── memory.py
└── config.py (.env loading)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone / Navigate to the project
```bash
cd CHATBOT
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
Open `.env` and replace the placeholder:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Run the chatbot
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 File Guide

| File | Purpose |
|---|---|
| `config.py` | Central configuration, loads `.env` |
| `llm_setup.py` | Creates ChatOpenAI and Embedding instances |
| `prompts.py` | Chat and RAG prompt templates |
| `memory.py` | Per-session conversation history management |
| `chatbot.py` | General chat chain with streaming |
| `rag.py` | PDF loading, FAISS indexing, RAG chain |
| `app.py` | Streamlit UI entry point |

---

## 🧠 How Memory Works in LangChain

### The Modern Approach: `RunnableWithMessageHistory`

LangChain's memory has evolved from tightly-coupled `Memory` objects to a clean two-part system:

**1. Storage Layer — `ChatMessageHistory`**
A simple list of `HumanMessage` and `AIMessage` objects, keyed by session ID.
```python
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

**2. Integration Layer — `RunnableWithMessageHistory`**
A wrapper that automatically:
- Loads the session's history before each chain call
- Injects it into the prompt's `MessagesPlaceholder`
- Saves the new messages after the call

```python
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### Memory Strategies

| Strategy | Pros | Cons |
|---|---|---|
| **Buffer** (used here) | Simple, full history | Token usage grows unbounded |
| **Window** (last K turns) | Predictable token usage | Loses old context |
| **Summary** | Retains essence, saves tokens | Extra LLM calls to summarize |
| **Vector Store** | Retrieves relevant past messages | Complex setup |

---

## 🚀 Improvements & Scaling Ideas

1. **Persistent Memory**: Replace in-memory dict with Redis or PostgreSQL
2. **Streaming Summaries**: Add `ConversationSummaryBufferMemory` for long chats
3. **Multi-document RAG**: Support multiple PDFs with metadata filtering
4. **Authentication**: Add user login with per-user session management
5. **LangGraph**: Migrate to stateful graph architecture for complex agent flows
6. **Caching**: Add LangChain caching to avoid duplicate API calls
7. **Monitoring**: Integrate LangSmith for tracing and debugging
8. **Deployment**: Containerize with Docker, deploy to AWS/GCP/Azure

---

## 📄 License

MIT — use freely for learning and production.
