import streamlit as st
# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Madrix - Personal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import EMBEDDER_MODEL, INDEX_PATH, DOCS_PATH, MODEL_NAME

# ---------- LOAD MODELS & DATA ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDER_MODEL)

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)

@st.cache_data
def load_docs():
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel(MODEL_NAME)

embedder = load_embedder()
index = load_index()
docs = load_docs()
model = load_gemini()

# ---------- RAG ANSWERING FUNCTION ----------
def rag_answer(query: str):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=3)
    context = "\n".join([docs[i] for i in indices[0]])

    prompt = f"""
You are Madrix, an assistant on Sarmad Rj's portfolio site.
Always answer in third person when asked about Sarmad Rj.
Use the given context to answer the question.

Context:
{context}

Question: {query}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- HEADER ----------
st.markdown(
    """
    <style>
    .centered {text-align: center;}
    .chat-bubble-user {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .chat-bubble-bot {
        background-color: #ECECEC;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import base64
import streamlit as st

def show_centered_logo(path: str, width: int = 100):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; width:97%;">
            <img src="data:image/png;base64,{b64}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True,
    )

# usage
show_centered_logo("logo.png", width=100)




st.markdown("<h1 class='centered'>MADRIX</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered'>Your personal AI assistant, trained on Sarmad Rj’s projects, skills, and achievements.</p>", unsafe_allow_html=True)

# ---------- CHAT ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat in reverse order (newest at bottom)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
query = st.chat_input("Ask Madrix something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    answer = rag_answer(query)
    st.session_state.messages.append({"role": "bot", "content": answer})
    st.rerun()

