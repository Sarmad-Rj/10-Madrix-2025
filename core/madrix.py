# Imports
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st

from .config import EMBEDDER_MODEL, INDEX_PATH, DOCS_PATH, MODEL_NAME, GEMINI_API_KEYS
from .prompt_templates import build_prompt

# --- Lazy-loaded singletons ---
_embedder = None
_index = None
_docs = None
_model = None

# --- Model Initialization ---
def init_models(force_reset=False):
    """
    Load and initialize models and data.
    If force_reset=True, _model will be reinitialized (useful for key rotation).
    """
    global _embedder, _index, _docs, _model

    if force_reset:
        _model = None

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_MODEL)

    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))

    if _docs is None:
        with open(DOCS_PATH, "rb") as f:
            _docs = pickle.load(f)

    if _model is None:
        genai.configure(api_key=GEMINI_API_KEYS[0])
        _model = genai.GenerativeModel(MODEL_NAME)

    return _embedder, _index, _docs, _model

# --- Reset model helper ---
def reset_model():
    global _model
    _model = None

# --- Helper for conversation history ---
def _history_text(messages, last_n=4):
    if not messages:
        return ""
    tail = messages[-last_n:]
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Madrix'}: {m['content']}"
        for m in tail
    )

# --- Core RAG answer ---
def rag_answer(query: str, mood: str, messages=None, top_k=3, last_n=4) -> str:
    embedder, index, docs, model = init_models()
    history = _history_text(messages or [], last_n=last_n)

    # Encode query and retrieve top-k similar documents
    q_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, k=top_k)
    context = "\n".join([docs[i] for i in indices[0]])

    # Build prompt and generate response
    prompt = build_prompt(mood=mood, history=history, context=context, query=query)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# --- Safe wrapper with 2-key rotation ---
def safe_rag_answer(query, mood, messages=None, top_k=3, last_n=4):
    """
    Rotate through Gemini API keys if quota is exceeded.
    Returns only the final answer (no key info displayed in Streamlit).
    """
    if "last_successful_key" not in st.session_state:
        st.session_state.last_successful_key = 0

    num_keys = len(GEMINI_API_KEYS)
    start_idx = st.session_state.last_successful_key

    for i in range(num_keys):
        idx = (start_idx + i) % num_keys

        # Force re-init model with current key
        reset_model()
        genai.configure(api_key=GEMINI_API_KEYS[idx])
        global _model
        _model = genai.GenerativeModel(MODEL_NAME)

        try:
            response = rag_answer(query, mood=mood, messages=messages, top_k=top_k, last_n=last_n)
            st.session_state.last_successful_key = idx
            return response  # only answer is returned
        except ResourceExhausted:
            continue

    return "⚠️ Daily API quota exceeded. Please try again tomorrow."
