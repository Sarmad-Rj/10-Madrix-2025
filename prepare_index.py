import pickle
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDER_MODEL, DATA_DIR, INDEX_PATH, DOCS_PATH
import os

# Load embedding model
embedder = SentenceTransformer(EMBEDDER_MODEL)

# Load personal facts from file
facts_file = os.path.join(DATA_DIR, "personal_facts.txt")
if not os.path.exists(facts_file):
    raise FileNotFoundError(f"❌ No personal_facts.txt found in {DATA_DIR}. Please create it and add your facts.")

with open(facts_file, "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

if not docs:
    raise ValueError("❌ personal_facts.txt is empty. Please add your info.")

# Convert to embeddings
embeddings = embedder.encode(docs, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, INDEX_PATH)

# Save docs list
with open(DOCS_PATH, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ Index built and saved to {INDEX_PATH}")
print(f"✅ Docs saved to {DOCS_PATH}")
