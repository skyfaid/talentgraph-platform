"""
Configuration settings for the CV Ranking System.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "resumes"

# LLM configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required. Please set it before running the application.")
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEMPERATURE = 0.1
GROQ_TOP_P = 0.9
GROQ_MAX_TOKENS = 1024
GROQ_SEED = 1

# Ranking configuration
SEMANTIC_WEIGHT = 0.3  # Weight for semantic similarity (30%)
LLM_WEIGHT = 0.7  # Weight for LLM score (70%)
DEFAULT_TOP_K = 5  # Default number of top candidates to return

# Text processing configuration
MIN_RESUME_LENGTH = 50  # Minimum character length for a valid resume
RESUME_HEAD_CHARS = 6000  # Characters to keep from start of resume for LLM
RESUME_TAIL_CHARS = 3000  # Characters to keep from end of resume for LLM

# Retriever configuration
RETRIEVER_K = 5  # Number of documents to retrieve in initial search
RANKING_SEARCH_MULTIPLIER = 2  # Multiply top_k by this for initial search

