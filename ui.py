import streamlit as st
from rag import build_index, retrieve
from model import load_model

st.title("Sarmad Rj - Personal Chatbot")

docs = [
    "Sarmad Rj is studying at COMSATS University in Pakistan.",
    "Sarmad Rj is skilled in Python, Java, and web development.",
    "Sarmad Rj created a school management system using Java Swing and Oracle SQL."
]

embedder, index = build_index(docs)
generator = load_model()

query = st.text_input("Ask a question about Sarmad Rj:")

if query:
    context = retrieve(query, docs, embedder, index, k=1)[0]
    prompt = f"""You are an assistant on Sarmad Rj's portfolio site.
Always answer in third person about Sarmad Rj.
Use the given context to answer the question.

Context: {context}
Question: {query}
Answer:"""
    result = generator(prompt)[0]['generated_text']
    st.write(result.split("Answer:")[-1].strip())
