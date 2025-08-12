# Install dependencies (run in terminal or notebook cell)
# pip install faiss-cpu sentence-transformers transformers accelerate

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Load embedding model (for semantic search)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Your personal facts
docs = [
    "Sarmad Rj is studying at COMSATS University in Pakistan.",
    "Sarmad Rj is skilled in Python, Java, and web development.",
    "Sarmad Rj created a school management system using Java Swing and Oracle SQL."
]

# 3. Create FAISS index
embeddings = embedder.encode(docs, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Load TinyLLaMA model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

# 5. Function to answer
def rag_answer(query):
    # Search FAISS
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=1)
    context = docs[indices[0][0]]

    # Prompt the model
    prompt = f"""You are an assistant on Sarmad Rj's portfolio site.
Always answer in third person about Sarmad Rj.
Use the given context to answer the question.

Context: {context}
Question: {query}
Answer:"""

    result = generator(prompt)[0]['generated_text']
    return result.split("Answer:")[-1].strip()

# 6. Try it
print(rag_answer("is sarmad currently studying?"))
print(rag_answer("What programming languages does he know?"))
