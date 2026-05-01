"""
memory.py — Conversation Memory Management
============================================
Manages per-session conversation history using LangChain's
ChatMessageHistory. This is the modern replacement for the
deprecated ConversationBufferMemory.

HOW MEMORY WORKS IN LANGCHAIN (Educational)
--------------------------------------------
LangChain's memory system has evolved significantly:

1. **Legacy (deprecated):** ConversationBufferMemory stored messages
   in a simple list and was tightly coupled to specific chain types
   (like ConversationChain). It was simple but inflexible.

2. **Modern (current):** The recommended approach uses two pieces:
   a) ChatMessageHistory — a simple store of HumanMessage/AIMessage
      objects, keyed by session ID.
   b) RunnableWithMessageHistory — a wrapper that automatically
      injects the chat history into your LCEL chain before each call
      and saves the new messages after.

   This separation means:
   - The *storage* (ChatMessageHistory) is independent of the *chain*
   - You can swap in Redis, PostgreSQL, or any backend without
     changing your chain logic
   - Multiple sessions are trivially supported via session IDs

MEMORY STRATEGIES:
   - BufferMemory: Store everything (what we use — simple & effective)
   - WindowMemory: Store only last K exchanges (saves tokens)
   - SummaryMemory: Periodically summarize older messages (saves tokens
     while retaining context)
   - Vector Store Memory: Embed messages and retrieve relevant ones
     (best for very long conversations)

For this project, we use a simple in-memory buffer. In production,
you'd replace the dict-based store with Redis or PostgreSQL.
"""

from langchain_community.chat_message_histories import ChatMessageHistory


# ── In-Memory Session Store ─────────────────────────────────
# Maps session IDs to their ChatMessageHistory instances.
# In production, replace this with a persistent store like:
#   - RedisChatMessageHistory
#   - PostgresChatMessageHistory
#   - CosmosDBChatMessageHistory

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Retrieve or create a ChatMessageHistory for the given session.

    This function is passed to RunnableWithMessageHistory as a
    callback. It's called automatically before each chain invocation
    to load the conversation history.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        ChatMessageHistory: The message history for this session.
    """
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session(session_id: str) -> None:
    """
    Clear the conversation history for a given session.
    Used when the user clicks "Clear Chat" in the UI.

    Args:
        session_id: The session to clear.
    """
    if session_id in _session_store:
        _session_store[session_id].clear()


def list_sessions() -> list[str]:
    """
    List all active session IDs.
    Useful for debugging or building a session selector in the UI.

    Returns:
        List of active session ID strings.
    """
    return list(_session_store.keys())


def get_message_count(session_id: str) -> int:
    """
    Get the number of messages in a session's history.

    Args:
        session_id: The session to check.

    Returns:
        Number of messages (both human and AI).
    """
    if session_id in _session_store:
        return len(_session_store[session_id].messages)
    return 0
