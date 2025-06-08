from openai import OpenAIEmbeddings  # or use SentenceTransformer
import faiss
import numpy as np

def get_embeddings(text_chunks, model):
    embeddings = model.embed_documents(text_chunks)
    return embeddings

def save_vector_store(text_chunks, embeddings, file_path="data/vector_store.faiss"):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, file_path)
    return index
