"""
app.py — Streamlit Chat Interface
====================================
Polished chat UI with two modes: General Chat and PDF Q&A.
Run with: streamlit run app.py
"""

import streamlit as st
import uuid

from chatbot import chat_stream
from rag import load_and_split_pdf_from_bytes, create_vector_store, query_pdf_stream
from memory import clear_session, get_message_count
import config


# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="🤖 LangChain Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ─────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500&display=swap');

    /* Global font & dark theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    /* Force deep black background for the main app container */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #050505 !important;
    }

    /* Main header styling with neon glow */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        animation: pulseGlow 3s infinite alternate;
    }
    .main-header h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00ffcc;
        text-shadow: 0 0 5px #00ffcc, 0 0 15px #00ffcc, 0 0 30px #00ffcc;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: 2px;
    }
    .main-header p {
        color: #ff00ff;
        font-size: 1rem;
        text-shadow: 0 0 5px #ff00ff;
        letter-spacing: 1px;
    }

    @keyframes pulseGlow {
        0% { text-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc; }
        100% { text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc, 0 0 40px #00ffcc; }
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #00ffcc;
        box-shadow: 2px 0 15px rgba(0, 255, 204, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ff00ff;
        text-shadow: 0 0 5px #ff00ff;
    }

    /* Streamlit input widgets (buttons, file uploader) hover effects */
    button[kind="secondary"] {
        border: 1px solid #ff00ff !important;
        color: #ff00ff !important;
        background: transparent !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(255, 0, 255, 0.1) !important;
        box-shadow: 0 0 10px #ff00ff, inset 0 0 10px #ff00ff !important;
        transform: translateY(-2px);
    }

    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .mode-chat {
        border: 1px solid #00ffcc;
        color: #00ffcc;
        box-shadow: 0 0 8px #00ffcc;
    }
    .mode-pdf {
        border: 1px solid #ff00ff;
        color: #ff00ff;
        box-shadow: 0 0 8px #ff00ff;
    }

    /* Status cards with interactive hover */
    .status-card {
        background: #111;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .status-card:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4);
        transform: translateY(-3px);
    }
    .status-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,255,204,0.1), transparent);
        transition: 0.5s;
    }
    .status-card:hover::before {
        left: 100%;
    }

    /* Chat messages animation & styling */
    .stChatMessage {
        animation: slideUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .stChatMessage:hover {
        border-color: rgba(0, 255, 204, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Chat input box neon focus */
    [data-testid="stChatInput"] {
        border: 1px solid #333 !important;
        background: #0a0a0a !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #00ffcc !important;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.5) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #444;
        font-size: 0.85rem;
        padding-top: 2rem;
        border-top: 1px dashed #333;
        margin-top: 2rem;
        font-family: 'Orbitron', sans-serif;
    }
    .footer span {
        color: #ff00ff;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "💬 General Chat"

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = 0


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Mode selector
    mode = st.radio(
        "Chat Mode",
        ["💬 General Chat", "📄 PDF Q&A"],
        index=0 if st.session_state.chat_mode == "💬 General Chat" else 1,
        help="Switch between general conversation and PDF document Q&A.",
    )
    st.session_state.chat_mode = mode

    st.markdown("---")

    # PDF uploader (shown only in PDF mode)
    if mode == "📄 PDF Q&A":
        st.markdown("### 📎 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Max size: {config.MAX_UPLOAD_SIZE_MB} MB",
        )

        if uploaded_file is not None:
            # Only process if it's a new file
            if uploaded_file.name != st.session_state.pdf_name:
                with st.spinner("📖 Processing PDF..."):
                    try:
                        file_bytes = uploaded_file.read()
                        chunks = load_and_split_pdf_from_bytes(
                            file_bytes, uploaded_file.name
                        )
                        vector_store = create_vector_store(chunks)
                        st.session_state.retriever = vector_store.as_retriever(
                            search_kwargs={"k": config.MAX_RETRIEVAL_RESULTS}
                        )
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.pdf_chunks = len(chunks)
                        st.success(f"✅ Loaded **{uploaded_file.name}**")
                    except Exception as e:
                        st.error(f"❌ Failed to process PDF: {e}")

        # Show PDF info
        if st.session_state.pdf_name:
            st.markdown(
                f'<div class="status-card">'
                f"📄 **{st.session_state.pdf_name}**<br>"
                f"📦 {st.session_state.pdf_chunks} chunks indexed"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Session info
    st.markdown("### 📊 Session Info")
    msg_count = len(st.session_state.messages)
    st.markdown(f"🆔 Session: `{st.session_state.session_id}`")
    st.markdown(f"💬 Messages: **{msg_count}**")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    # New session button
    if st.button("➕ New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────
# Header
mode_class = "mode-chat" if mode == "💬 General Chat" else "mode-pdf"
mode_label = "General Chat" if mode == "💬 General Chat" else "PDF Q&A"

st.markdown(
    f"""
    <div class="main-header">
        <h1>🤖 LangChain Chatbot</h1>
        <p>Powered by Gemini &bull; <span class="mode-badge {mode_class}">{mode_label}</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# API key check
if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your-google-api-key-here":
    st.error(
        "🔑 **Google API key not configured!**\n\n"
        "1. Open `.env` in the project root\n"
        "2. Set `GOOGLE_API_KEY=your-key-here`\n"
        "3. Restart the app with `streamlit run app.py`"
    )
    st.stop()

# PDF mode warning if no doc uploaded
if mode == "📄 PDF Q&A" and st.session_state.retriever is None:
    st.warning("📎 Please upload a PDF document in the sidebar to get started.")

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        try:
            if mode == "💬 General Chat":
                # Stream general chat response
                response = st.write_stream(
                    chat_stream(prompt, st.session_state.session_id)
                )
            else:
                # Stream PDF Q&A response
                if st.session_state.retriever is None:
                    response = "📎 Please upload a PDF document first using the sidebar."
                    st.markdown(response)
                else:
                    response = st.write_stream(
                        query_pdf_stream(
                            prompt,
                            st.session_state.retriever,
                            st.session_state.session_id,
                        )
                    )
        except Exception as e:
            response = f"❌ Error: {e}"
            st.error(response)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown(
    '<div class="footer">'
    "Built with LangChain &bull; Gemini &bull; Streamlit &bull; FAISS"
    "</div>",
    unsafe_allow_html=True,
)
