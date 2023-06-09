import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


embedding_dim = 384
index_file_path = 'faiss_index.bin'
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index(index_file_path)

def query_faiss_index(query, top_k=5):
    query_embedding = model.encode([query])[0]
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Perform a search in the index
    D, I = index.search(query_embedding, top_k)
    return D, I

def run_query_chain(query):
    D, I = query_faiss_index(query)
    # Here you would typically take the IDs of the top documents (contained in I)
    # and fetch these documents from your database.
    # Then you would use these documents to answer the query.

    # For now, let's just print the indices of the documents and their distances
    for distances, indices in zip(D, I):
        for distance, index in zip(distances, indices):
            print(f"Document index: {index}, Distance: {distance}")

if __name__ == "__main__":
    query = "Tell me about gainwell?"
    run_query_chain(query)
