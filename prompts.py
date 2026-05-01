"""
prompts.py — Prompt Templates
===============================
Defines reusable prompt templates for the chatbot.
Uses LangChain's ChatPromptTemplate with MessagesPlaceholder
to support conversation history injection.

Two templates are provided:
  1. CHAT_PROMPT      — General conversational chatbot
  2. RAG_PROMPT       — PDF document question-answering
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ── General Chat Prompt ─────────────────────────────────────
# This prompt powers the general-purpose conversational mode.
# The {history} placeholder is automatically filled by
# RunnableWithMessageHistory with past conversation turns.

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful, friendly, and knowledgeable AI assistant. "
        "Follow these guidelines:\n"
        "• Provide clear, concise, and accurate answers.\n"
        "• If you don't know something, say so honestly.\n"
        "• Use markdown formatting when it improves readability.\n"
        "• Remember and reference earlier parts of our conversation "
        "when relevant.\n"
        "• Be conversational but professional."
    ),
    # This placeholder is where conversation history gets injected
    MessagesPlaceholder(variable_name="history"),
    # The current user message
    ("human", "{input}"),
])


# ── RAG (Retrieval-Augmented Generation) Prompt ─────────────
# Used when answering questions about uploaded PDF documents.
# The {context} variable is filled with relevant text chunks
# retrieved from the FAISS vector store.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant specialized in answering "
        "questions about documents. Use the following retrieved context "
        "to answer the user's question.\n\n"
        "RULES:\n"
        "• Base your answer ONLY on the provided context.\n"
        "• If the context doesn't contain enough information to answer, "
        "say: \"I couldn't find that information in the uploaded document.\"\n"
        "• Quote or reference specific parts of the document when possible.\n"
        "• Use markdown formatting for clarity.\n"
        "• Be concise but thorough.\n\n"
        "CONTEXT:\n"
        "─────────────────────────────────\n"
        "{context}\n"
        "─────────────────────────────────"
    ),
    # Conversation history for follow-up questions about the document
    MessagesPlaceholder(variable_name="history"),
    # The current user question
    ("human", "{input}"),
])
