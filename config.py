import os
import google.generativeai as genai

# 1. API KEY CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API key not set. Please set it in config.py or as an environment variable.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# 2. MODEL SETTINGS
MODEL_NAME = "gemini-2.0-flash-exp"

# 3. EMBEDDING SETTINGS
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "embeddings/faiss_index.bin"
DOCS_PATH = "embeddings/docs_list.pkl"
DATA_DIR = "data"

# Create required folders if missing
os.makedirs("embeddings", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
