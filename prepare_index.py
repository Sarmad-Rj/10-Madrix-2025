import pickle
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDER_MODEL, INDEX_PATH, DOCS_PATH

# Load embedding model
embedder = SentenceTransformer(EMBEDDER_MODEL)

# Load all your personal facts from multiple files or lists
# Example: 3 text files
file_paths = [
    "data/personal_facts.txt",
    "data/projects.txt",
    "data/skills.txt",
    "data/achievements.txt"
]

docs = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            fact = line.strip()
            if fact:
                docs.append(fact)

# Convert documents to embeddings
embeddings = embedder.encode(docs, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and documents
faiss.write_index(index, INDEX_PATH)
with open(DOCS_PATH, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ Index built with {len(docs)} facts from {len(file_paths)} files.")
