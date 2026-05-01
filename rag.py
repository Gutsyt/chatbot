"""
rag.py — PDF Document Question-Answering (RAG)
================================================
Implements Retrieval-Augmented Generation for uploaded PDFs.

PIPELINE:
    PDF → PyPDFLoader → TextSplitter → Embeddings → FAISS → Retriever → RAG Chain → Answer
"""

import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from llm_setup import get_chat_llm, get_embeddings
from prompts import RAG_PROMPT
from memory import get_session_history
import config


def load_and_split_pdf(file_path: str) -> list[Document]:
    """Load a PDF and split into overlapping text chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise ValueError("The PDF is empty or contains no extractable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def load_and_split_pdf_from_bytes(file_bytes: bytes, file_name: str) -> list[Document]:
    """Load a PDF from raw bytes (Streamlit uploader) and split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="chatbot_") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks = load_and_split_pdf(tmp_path)
        for chunk in chunks:
            chunk.metadata["source"] = file_name
        return chunks
    finally:
        os.unlink(tmp_path)


def create_vector_store(documents: list[Document]) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    if not documents:
        raise ValueError("No documents provided to create vector store.")
    embeddings = get_embeddings()
    return FAISS.from_documents(documents=documents, embedding=embeddings)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string with metadata."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Chunk {i} | Page {page} | {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_rag_chain(retriever):
    """Build the RAG LCEL chain: retriever → prompt → LLM → parser."""
    llm = get_chat_llm()
    chain = (
        {
            "context": lambda x: _format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
            "history": lambda x: x["history"],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def get_conversational_rag_chain(retriever):
    """Wrap RAG chain with RunnableWithMessageHistory for multi-turn Q&A."""
    base_chain = get_rag_chain(retriever)
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def query_pdf(question: str, retriever, session_id: str = None) -> str:
    """Ask a question about an uploaded PDF. Main entry point for PDF Q&A mode."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if retriever is None:
        raise ValueError("No document uploaded. Please upload a PDF first.")
    if session_id is None:
        session_id = config.DEFAULT_SESSION_ID

    try:
        chain = get_conversational_rag_chain(retriever)
        return chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            raise ConnectionError("🔑 Invalid API key. Check your GOOGLE_API_KEY.") from e
        elif "rate limit" in error_msg:
            raise ConnectionError("⏳ Rate limit exceeded. Wait and try again.") from e
        else:
            raise RuntimeError(f"❌ Error querying document: {e}") from e


def query_pdf_stream(question: str, retriever, session_id: str = None):
    """Stream a response about the PDF token by token."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if retriever is None:
        raise ValueError("No document uploaded. Please upload a PDF first.")
    if session_id is None:
        session_id = config.DEFAULT_SESSION_ID

    try:
        chain = get_conversational_rag_chain(retriever)
        for chunk in chain.stream(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk
    except Exception as e:
        yield f"\n\n❌ Error: {e}"
