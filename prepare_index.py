import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
EMBEDDER_NAME = "all-MiniLM-L6-v2"
OUTPUT_INDEX = "embeddings/faiss_index.bin"
OUTPUT_DOCS = "embeddings/docs_list.pkl"

# Load embedding model
embedder = SentenceTransformer(EMBEDDER_NAME)

# Read all text files
docs = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            for line in f:
                fact = line.strip()
                if fact:
                    docs.append(fact)

# Convert to embeddings
embeddings = embedder.encode(docs, convert_to_numpy=True)
dim = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index and docs list
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, OUTPUT_INDEX)
with open(OUTPUT_DOCS, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ Index built with {len(docs)} facts.")
