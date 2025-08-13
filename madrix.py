import pickle
import faiss
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
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=3)  # Retrieve top 3 matches
    context = "\n".join([docs[i] for i in indices[0]])

    prompt = f"""
You are Madrix, an assistant on Sarmad Rj's portfolio site.
Always answer in third person about Sarmad Rj.
Use the given context to answer the question.

Context:
{context}

Question: {query}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    print("💬 Madrix is ready! Ask your questions (type 'exit' to quit).")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = rag_answer(query)
        print(f"Madrix: {answer}")
