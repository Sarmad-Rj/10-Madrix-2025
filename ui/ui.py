# Imports
import streamlit as st
from pathlib import Path
import base64
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.madrix import safe_rag_answer, init_models

# --- Page Configuration ---
st.set_page_config(
    page_title="Madrix - Personal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- Styling (chat bubbles) ---
st.markdown(
    """
    <style>
    .centered {text-align: center;}
    .chat-bubble-user {
        background-color: #DCF8C6;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .chat-bubble-bot {
        background-color: #ECECEC;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    @media (prefers-color-scheme: dark) {
        .chat-bubble-user { background-color: #2e7d32; color: #ffffff; }
        .chat-bubble-bot { background-color: #424242; color: #ffffff; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helpers ---
def show_centered_logo(path: str, width: int = 100):
    """Display a centered logo image."""
    p = Path(path)
    if not p.exists():
        return
    b64 = base64.b64encode(p.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; width:97%;">
            <img src="data:image/png;base64,{b64}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- App State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood" not in st.session_state:
    st.session_state.mood = "friendly"

# --- Sidebar Settings ---
st.sidebar.header("⚙ Settings")
st.session_state.mood = st.sidebar.selectbox(
    "Mood", ["friendly", "rude"], index=(0 if st.session_state.mood == "friendly" else 1)
)
memory_n = 3
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# --- Header ---
show_centered_logo("logo/logo.png", width=100)
st.markdown("<h1 class='centered'>MADRIX</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='centered'>Your AI assistant, trained on Sarmad Rj’s projects, skills, and achievements.</p>",
    unsafe_allow_html=True
)

# --- Initialize Models ---
init_models()

# --- Render Chat History ---
for msg in st.session_state.messages:
    klass = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f"<div class='{klass}'>{msg['content']}</div>", unsafe_allow_html=True)

# --- User Input ---
query = st.chat_input("Ask Madrix something...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        # Only the final answer is returned; key info is internal
        answer = safe_rag_answer(
            query,
            mood=st.session_state.mood,
            messages=st.session_state.messages,
            last_n=memory_n
        )

    st.session_state.messages.append({"role": "bot", "content": answer})
    st.rerun()
