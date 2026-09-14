"""Central application configuration.

Values come from the environment (a project-root .env file when present)
or from defaults defined here. No secrets are hardcoded in source files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_DIR = DATA_DIR / "faiss"
CHROMA_DIR = DATA_DIR / "chroma"  # Archived legacy storage

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

# sentence-transformer model used to embed documents and queries.
# Empty means "use the application default" (all-MiniLM-L6-v2).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

# Context builder settings (Milestone 7.1)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "10"))
