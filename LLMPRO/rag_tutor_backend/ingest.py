# ingest.py
import os
import json
import uuid
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer

# Optional OCR support
from pdf2image import convert_from_path
import pytesseract

# Initialize your embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_DIR = "data/uploads"
CHUNKS_FILE = "data/chunks.json"
FAISS_DIR = "data/faiss_index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

def extract_text(pdf_path):
    """Try extracting text using pdfplumber first. If empty, use OCR."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("pdfplumber failed:", e)

    # If no text extracted, try OCR
    if len(text.strip()) == 0:
        print("No text found with pdfplumber. Using OCR...")
        pages = convert_from_path(pdf_path)
        for page in pages:
            text += pytesseract.image_to_string(page)

    if len(text.strip()) == 0:
        raise ValueError("No text could be extracted from the PDF.")
    return text

def save_chunks(chunks):
    """Save chunks to JSON for reference."""
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

def ingest_textbook(pdf_path, chunk_size=500):
    """Process PDF, create embeddings, and store in FAISS."""
    text = extract_text(pdf_path)
    print(f"Extracted text length: {len(text)} characters")

    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Number of chunks created: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No chunks created from text.")

    save_chunks(chunks)

    # Generate embeddings
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty!")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors of dimension {dim}")

    # Save FAISS index
    faiss_file = os.path.join(FAISS_DIR, "index.faiss")
    faiss.write_index(index, faiss_file)
    print(f"FAISS index saved at {faiss_file}")

    return chunks, index

