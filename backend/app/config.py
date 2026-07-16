import os
from pathlib import Path
from dotenv import load_dotenv

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from the backend directory
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Central configuration settings for the Multimodal RAG Assistant backend."""

    # --- API Keys ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SAMBANOVA_API_KEY: str = os.getenv("SAMBANOVA_API_KEY", "")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")

    # --- LLM Settings ---
    LLM_PROVIDER: str = "none"
    LLM_MODEL: str = ""

    # --- Directories ---
    INPUT_DIR: Path = BASE_DIR / os.getenv("INPUT_DIR", "data_sandbox/input_manuals")
    OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "data_sandbox/processed_markdown")

    # --- Embeddings ---
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # --- Qdrant Vector Store ---
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "manuals")
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "data_sandbox/qdrant_db")

    # --- RAG Parameters ---
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.0
    RRF_HIGH_THRESHOLD: float = 0.025  # Matches well in both dense and sparse or top 1
    RRF_LOW_THRESHOLD: float = 0.015   # Below this means it's ranked low in just one list

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # --- Phase 2 Limits ---
    MAX_CLARIFICATION_ATTEMPTS: int = 2
    MAX_RETRIEVAL_RETRIES: int = 1

    def __init__(self):
        # Auto-detect which provider to use (Groq preferred)
        if self.GROQ_API_KEY:
            self.LLM_PROVIDER = "groq"
            self.LLM_MODEL = "llama-3.1-8b-instant"
        elif self.SAMBANOVA_API_KEY:
            self.LLM_PROVIDER = "sambanova"
            self.LLM_MODEL = "Meta-Llama-3.1-8B-Instruct"


settings = Settings()

# Module-level variable mapping for backward compatibility
GROQ_API_KEY = settings.GROQ_API_KEY
SAMBANOVA_API_KEY = settings.SAMBANOVA_API_KEY
SARVAM_API_KEY = settings.SARVAM_API_KEY
LLM_PROVIDER = settings.LLM_PROVIDER
LLM_MODEL = settings.LLM_MODEL
EMBED_MODEL = settings.EMBED_MODEL
QDRANT_COLLECTION = settings.QDRANT_COLLECTION
QDRANT_PATH = settings.QDRANT_PATH
TOP_K = settings.TOP_K
SCORE_THRESHOLD = settings.SCORE_THRESHOLD
RRF_HIGH_THRESHOLD = settings.RRF_HIGH_THRESHOLD
RRF_LOW_THRESHOLD = settings.RRF_LOW_THRESHOLD
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
MAX_CLARIFICATION_ATTEMPTS = settings.MAX_CLARIFICATION_ATTEMPTS
MAX_RETRIEVAL_RETRIES = settings.MAX_RETRIEVAL_RETRIES

