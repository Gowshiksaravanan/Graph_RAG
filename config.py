import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

PROJECT_DIR = Path(__file__).parent
PROMPTS_DIR = PROJECT_DIR / "prompts"

KNOWN_MODEL_LIMITS = {
    "gpt-4o":       {"context_window": 128_000,   "doc_budget": 90_000},
    "gpt-4o-mini":  {"context_window": 128_000,   "doc_budget": 90_000},
    "gpt-4.1":      {"context_window": 1_000_000, "doc_budget": 900_000},
    "gpt-4.1-mini": {"context_window": 1_000_000, "doc_budget": 900_000},
    "gpt-4.1-nano": {"context_window": 1_000_000, "doc_budget": 900_000},
}

TIKTOKEN_ENCODING = "cl100k_base"
BATCH_TOKEN_BUDGET = 30_000
CHUNK_TOKEN_LIMIT = 2_000
ACCEPTED_FILE_TYPES = ["txt", "pdf", "docx"]
