import pickle
import faiss

INDEX_PATH = "embeddings/faiss_index.bin"
DOCS_PATH = "embeddings/docs_list.pkl"

# Load FAISS index & docs
index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "rb") as f:
    docs = pickle.load(f)
