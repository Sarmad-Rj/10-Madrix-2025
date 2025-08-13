import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import EMBEDDER_MODEL, INDEX_PATH, DOCS_PATH, MODEL_NAME

# Load embedding model
embedder = SentenceTransformer(EMBEDDER_MODEL)

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load documents
with open(DOCS_PATH, "rb") as f:
    docs = pickle.load(f)

# Load Gemini model
model = genai.GenerativeModel(MODEL_NAME)


def rag_answer(query: str):
    # Encode query
    query_vec = embedder.encode([query], convert_to_numpy=True)

    # Search top 1 relevant document
    distances, indices = index.search(query_vec, k=1)
    context = docs[indices[0][0]]

    # Prepare prompt
    prompt = f"""
You are Madrix, an assistant on Sarmad Rj's portfolio site.
Always answer in third person about Sarmad Rj.
Use the given context to answer the question.

Context: {context}
Question: {query}
Answer:
"""
    # Get Gemini's response
    response = model.generate_content(prompt)
    return response.text.strip()


# Test
if __name__ == "__main__":
    print(rag_answer("Is Sarmad currently studying?"))
    print(rag_answer("What programming languages does he know?"))
