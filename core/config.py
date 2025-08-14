from pathlib import Path
import os

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EMB_DIR = BASE_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DOCS_PATH = EMB_DIR / "docs_list.pkl"
INDEX_PATH = EMB_DIR / "faiss_index.bin"

# --- Model names ---
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# --- Gemini API keys (rotation) ---
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
]
_current_key_index = 0

def get_gemini_api_key() -> str:
    """
    Return the currently active Gemini API key.
    Raises error if no keys are set.
    """
    global _current_key_index
    if not GEMINI_API_KEYS or GEMINI_API_KEYS[_current_key_index] is None:
        raise ValueError("No valid GEMINI_API_KEY set in environment variables or Streamlit secrets.")
    return GEMINI_API_KEYS[_current_key_index]

def rotate_gemini_key() -> str:
    """
    Switch to the next available API key in the list.
    Returns the new active key.
    """
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
    return GEMINI_API_KEYS[_current_key_index]
