"""
llm_setup.py — LLM & Embedding Initialization
===============================================
Provides factory functions for creating LangChain LLM and
embedding instances. All configuration is pulled from config.py
so callers don't need to worry about API keys or model names.
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import config


def get_chat_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return a configured ChatGoogleGenerativeAI instance.

    Returns:
        ChatGoogleGenerativeAI: Ready-to-use LLM instance.

    Raises:
        ValueError: If the API key is missing or invalid.
        ConnectionError: If the Google API is unreachable.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Please set it in your .env file."
        )

    try:
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
            google_api_key=config.GOOGLE_API_KEY,
        )
        return llm

    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize ChatGoogleGenerativeAI: {e}\n"
            "Check your API key and network connection."
        ) from e


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return a configured GoogleGenerativeAIEmbeddings instance.
    Used by the RAG pipeline to embed document chunks and queries.

    Returns:
        GoogleGenerativeAIEmbeddings: Ready-to-use embedding model.

    Raises:
        ValueError: If the API key is missing.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Please set it in your .env file."
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
        return embeddings

    except Exception as e:
        raise ConnectionError(
            f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}"
        ) from e
