"""Centralized configuration - API keys, paths, and retrieval parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# --- Paths ---
RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")
METADATA_DIR = str(PROJECT_ROOT / "data" / "metadata")
TEST_SET_DIR = str(PROJECT_ROOT / "data" / "test_set")
CHROMA_DB_DIR = str(PROJECT_ROOT / "chroma_db")
PROMPTS_DIR = str(PROJECT_ROOT / "prompts")
LOGS_DB_PATH = str(PROJECT_ROOT / "monitoring" / "logs.db")

EXTRACTED_PAGES_PATH = os.path.join(PROCESSED_DATA_DIR, "extracted_pages.json")
DEFAULT_CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "chunks_recursive.json")

# --- Embedding ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# --- LLM defaults ---
DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1000
TEMPERATURE = 0.1  # Low temperature for factual clinical answers

# Approx pricing (USD per token) for cost estimation only
MODEL_PRICING = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
    "claude-3-5-haiku-20241022": {"in": 0.80 / 1_000_000, "out": 4.00 / 1_000_000},
    "claude-3-haiku-20240307": {"in": 0.25 / 1_000_000, "out": 1.25 / 1_000_000},
}

# --- Retrieval ---
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# --- ChromaDB ---
COLLECTION_NAME = "clinical_guidelines"


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return an estimated USD cost for a request, or 0.0 if pricing is unknown."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return tokens_in * pricing["in"] + tokens_out * pricing["out"]
