"""
chatbot.py — Core Chatbot Logic
=================================
Builds the conversational chain using LCEL (LangChain Expression
Language) and wraps it with RunnableWithMessageHistory for
automatic conversation memory management.

ARCHITECTURE:
    User Input
        ↓
    RunnableWithMessageHistory (loads/saves history)
        ↓
    ChatPromptTemplate (formats system + history + input)
        ↓
    ChatGoogleGenerativeAI (generates response)
        ↓
    StrOutputParser (extracts text from AIMessage)
        ↓
    Response String
"""

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from llm_setup import get_chat_llm
from prompts import CHAT_PROMPT
from memory import get_session_history
import config


def get_chat_chain():
    """
    Build the base LCEL chat chain (without memory).

    The chain is composed using the pipe (|) operator:
        prompt → LLM → output parser

    Returns:
        RunnableSequence: The composed chain.
    """
    llm = get_chat_llm()

    # LCEL composition: prompt template → LLM → parse to string
    chain = CHAT_PROMPT | llm | StrOutputParser()
    return chain


def get_conversational_chain():
    """
    Wrap the base chain with RunnableWithMessageHistory to add
    automatic conversation memory.

    The wrapper:
    1. Calls get_session_history(session_id) to load past messages
    2. Injects them into the "history" placeholder in the prompt
    3. Runs the chain
    4. Saves the new human + AI messages to history

    Returns:
        RunnableWithMessageHistory: The memory-enabled chain.
    """
    base_chain = get_chat_chain()

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,          # Callback to load/create history
        input_messages_key="input",   # Maps to {input} in the prompt
        history_messages_key="history",  # Maps to MessagesPlaceholder("history")
    )
    return chain_with_history


def chat(user_input: str, session_id: str = None) -> str:
    """
    Send a message to the chatbot and get a response.

    This is the main entry point for the general chat mode.
    It handles the full flow: load history → format prompt →
    call LLM → save messages → return response.

    Args:
        user_input: The user's message text.
        session_id: Unique session identifier. Defaults to
                    config.DEFAULT_SESSION_ID.

    Returns:
        str: The chatbot's response text.

    Raises:
        ValueError: If user_input is empty.
        Exception: Re-raises API errors with helpful context.
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if session_id is None:
        session_id = config.DEFAULT_SESSION_ID

    try:
        chain = get_conversational_chain()

        # The config dict tells RunnableWithMessageHistory which
        # session to use for loading/saving history.
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return response

    except ValueError as e:
        raise e
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            raise ConnectionError(
                "🔑 Invalid API key. Please check your GOOGLE_API_KEY "
                "in the .env file."
            ) from e
        elif "rate limit" in error_msg:
            raise ConnectionError(
                "⏳ Rate limit exceeded. Please wait a moment and try again."
            ) from e
        elif "model" in error_msg and "not found" in error_msg:
            raise ConnectionError(
                f"🤖 Model '{config.LLM_MODEL}' not found. "
                "Check your config.py settings."
            ) from e
        else:
            raise RuntimeError(
                f"❌ An unexpected error occurred: {e}\n"
                "Please check your network connection and API key."
            ) from e


def chat_stream(user_input: str, session_id: str = None):
    """
    Stream a response from the chatbot token by token.

    Yields chunks of text as they arrive from the LLM, enabling
    real-time display in the Streamlit UI.

    Args:
        user_input: The user's message text.
        session_id: Unique session identifier.

    Yields:
        str: Chunks of the response text.
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if session_id is None:
        session_id = config.DEFAULT_SESSION_ID

    try:
        chain = get_conversational_chain()

        # .stream() yields chunks as they arrive from the API
        for chunk in chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk

    except Exception as e:
        yield f"\n\n❌ Error: {e}"
