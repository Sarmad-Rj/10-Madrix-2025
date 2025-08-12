# pip install faiss-cpu sentence-transformers google-generativeai

import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# 1. Set API Key (must be stored as environment variable)
#   Windows: setx GEMINI_API_KEY "your_api_key"
#   Mac/Linux: export GEMINI_API_KEY="your_api_key"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Your personal facts
docs = [
    "Sarmad Rj is studying at COMSATS University in Pakistan.",
    "Sarmad Rj is skilled in Python, Java, and web development.",
    "Sarmad Rj created a school management system using Java Swing and Oracle SQL."
]

# 4. Create FAISS index
embeddings = embedder.encode(docs, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Function to answer
def rag_answer(query):
    # Search in FAISS
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=1)
    context = docs[indices[0][0]]

    # Personality-enhanced system instructions
    system_prompt = """
You are Madrix — a helpful, confident, and slightly witty AI assistant that only talks about Sarmad Rj.
Always speak in the third person when referring to Sarmad.
Keep answers clear, accurate, and warm in tone.
Never reveal that you are an AI model — act as if you know Sarmad personally.
"""

    # Final prompt
    prompt = f"""{system_prompt}

Context: {context}
Question: {query}
Answer:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# 6. Test
print(rag_answer("is sarmad currently studying?"))
print(rag_answer("What programming languages does he know?"))

