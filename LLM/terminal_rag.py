import os
import sys
import argparse
import logging

# Suppress non-critical warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Prefer community loaders (installed via requirements)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Optional imports - chains may not be present in all langchain builds
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    HAS_CHAINS = True
except Exception:
    create_stuff_documents_chain = None
    create_retrieval_chain = None
    HAS_CHAINS = False

# --- DEFAULT CONFIGURATION ---
DEFAULT_DATA_PATH = "./documents"
DEFAULT_DB_FAISS_PATH = "./vectorstore"
DEFAULT_MODEL_NAME = "llama3"  # Ensure you ran 'ollama pull llama3' if using Ollama

def main():
    # 1. Check if document exists
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"❌ Error: Please put a PDF file inside the '{DATA_PATH}' folder.")
        return

    print("🔄 Processing PDF... (This allows Llama3 to read it)")
    
    # 2. Ingest Data (Load & Split)
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDF found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Vector Store
    print("🧠 Creating Memory (Vector Store)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and Save locally
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    # 4. Setup Llama 3 Connection
    print("🦙 Connecting to Llama 3...")
    llm = ChatOllama(model=MODEL_NAME)
    
    # Load the DB we just created
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 5. Define the "Description" Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant. 
    Read the following context from a document and provide a detailed summary and description of what this document is about.
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 6. Run the Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n" + "="*40)
    print("📝 GENERATING DOCUMENT DESCRIPTION...")
    print("="*40 + "\n")

    # Ask the specific question to trigger a summary
    response = retrieval_chain.invoke({"input": "Describe this document and summarize its key points."})

    # 7. Print Result
    print(response["answer"])
    print("\n" + "="*40)

if __name__ == "__main__":
    main()