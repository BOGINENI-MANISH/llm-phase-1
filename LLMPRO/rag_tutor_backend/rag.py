# rag.py
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import subprocess

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
CHUNKS_FILE = "data/chunks.json"
FAISS_FILE = "data/faiss_index/index.faiss"

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load FAISS index
index = faiss.read_index(FAISS_FILE)

def ask_question(query, top_k=3):
    """Return top-k relevant chunks for a query using FAISS embeddings."""
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    combined_context = "\n\n".join(results)
    
    # Optional: Call your LLM for final answer
    # Using Ollama (assuming you have it installed and a local model like llama2)
    try:
        process = subprocess.run(
            ["ollama", "generate", "llama2", combined_context, "-p", query],
            capture_output=True, text=True
        )
        answer = process.stdout.strip()
    except Exception as e:
        print("Ollama call failed:", e)
        answer = combined_context  # fallback: just return context

    return answer
