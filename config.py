"""
config.py — Central Configuration
==================================
Loads environment variables and exposes all tuneable settings
as module-level constants. Every other module imports from here
so there is a single source of truth.
"""

import os
import sys
from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────
# find_dotenv searches parent directories too, but we explicitly
# point at the project root to be deterministic.
load_dotenv()

# ── Google API Key ──────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-google-api-key-here":
    print(
        "\n[WARNING] GOOGLE_API_KEY is not set!\n"
        "   1. Open the .env file in the project root.\n"
        "   2. Replace 'your-google-api-key-here' with your real key.\n"
        "   3. Restart the application.\n"
    )
    # We don't sys.exit() here so Streamlit can still render an
    # error banner instead of just crashing silently.

# ── LLM Settings ────────────────────────────────────────────
LLM_MODEL: str = "gemini-2.5-flash"       # Default chat model
LLM_TEMPERATURE: float = 0.7         # Creativity (0 = deterministic, 1 = creative)
LLM_MAX_TOKENS: int = 1024           # Max tokens per response

# ── Embedding Settings ──────────────────────────────────────
EMBEDDING_MODEL: str = "gemini-embedding-001"  # Fast & cheap

# ── RAG / Document Processing ──────────────────────────────
CHUNK_SIZE: int = 1000               # Characters per text chunk
CHUNK_OVERLAP: int = 200             # Overlap between chunks for context continuity
MAX_RETRIEVAL_RESULTS: int = 4       # Number of chunks to retrieve per query
MAX_UPLOAD_SIZE_MB: int = 200        # Maximum PDF upload size

# ── Session Defaults ────────────────────────────────────────
DEFAULT_SESSION_ID: str = "default"  # Fallback session identifier
