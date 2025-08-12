from sentence_transformers import SentenceTransformer
import faiss

def build_index(docs):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embedder, index

def retrieve(query, docs, embedder, index, k=1):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k=k)
    return [docs[i] for i in indices[0]]
